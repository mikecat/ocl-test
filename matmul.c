#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include "CL/cl.h"

double get_time(void) {
	struct timeval tv;
	if (gettimeofday(&tv, NULL) != 0) {
		perror("gettimeofday failed");
		return 0;
	} else {
		return tv.tv_sec + tv.tv_usec * 1e-6;
	}
}

static const char* kernel_code =
"__kernel void matmul(__global float* a, __global float* b, __global float* out, int size) {\n"
"  int i = get_global_id(0);\n"
"  int j = get_global_id(1);\n"
"  int k;\n"
"  float ret = 0;\n"
"  if (i < size && j < size) {\n"
"    for (k = 0; k < size; k++) {\n"
"      ret += a[i * size + k] * b[k * size + j];\n"
"    }\n"
"    out[i * size + j] = ret;\n"
"  }\n"
"}\n"
;

void matmul_normal(const float* a, const float* b, float* out, int size) {
	int i, j, k;
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < size; i++) {
		for (j = 0; j < size; j++) {
			float ret = 0;
			for (k = 0; k < size; k++) {
				ret += a[i * size + k] * b[k * size + j];
			}
			out[i * size + j] = ret;
		}
	}
}

int calc_main(cl_platform_id platform, cl_device_id device, cl_context context,
int argc, char* argv[]) {
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_int error;

	int size = (argc > 0 ? atoi(argv[0]) : 1024);
	float *a, *b, *out, *out_normal;
	size_t matrix_data_size;
	cl_mem a_cl, b_cl, out_cl;
	size_t global_size[2], local_size[2] = {8, 8};
	double time_start, time_end, flop;
	int i;
	float diff_sum;

	if (size <= 0) {
		fputs("matrix size must be positive\n", stderr);
		return 1;
	}

	global_size[0] = ((size + local_size[0] - 1) / local_size[0]) * local_size[0];
	global_size[1] = ((size + local_size[1] - 1) / local_size[1]) * local_size[1];

	matrix_data_size = sizeof(float) * size * size;
	flop = 2.0 * size * size * size;
	a = malloc(matrix_data_size);
	b = malloc(matrix_data_size);
	out = malloc(matrix_data_size);
	out_normal = malloc(matrix_data_size);
	if (a == NULL || b == NULL || out == NULL || out_normal == NULL) {
		fputs("malloc failed\n", stderr);
		free(a); free(b); free(out); free(out_normal);
		return 1;
	}

#ifdef CL_VERSION_2_0
	queue = clCreateCommandQueueWithProperties(context, device, NULL, &error);
#else
	queue = clCreateCommandQueue(context, device, 0, &error);
#endif
	if (error != CL_SUCCESS) {
		fprintf(stderr, "clCreateCommandQueue failed: %d\n", (int)error);
		free(a); free(b); free(out); free(out_normal);
		return 1;
	}

	program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "clCreateProgramWithSource failed: %d\n", (int)error);
		free(a); free(b); free(out); free(out_normal);
		clReleaseCommandQueue(queue);
		return 1;
	}

	if ((error = clBuildProgram(program, 1, &device, "-w", NULL, NULL)) != CL_SUCCESS) {
		fprintf(stderr, "clBuildProgram failed: %d\n", (int)error);
		free(a); free(b); free(out); free(out_normal);
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		return 1;
	}

	kernel = clCreateKernel(program, "matmul", &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "clCreateKernel failed: %d\n", (int)error);
		free(a); free(b); free(out); free(out_normal);
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		return 1;
	}

	a_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_data_size, NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "clCreateBuffer for A failed: %d\n", (int)error);
		free(a); free(b); free(out); free(out_normal);
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		return 1;
	}
	b_cl = clCreateBuffer(context, CL_MEM_READ_ONLY, matrix_data_size, NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "clCreateBuffer for B failed: %d\n", (int)error);
		free(a); free(b); free(out); free(out_normal);
		clReleaseMemObject(a_cl);
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		return 1;
	}
	out_cl = clCreateBuffer(context, CL_MEM_WRITE_ONLY, matrix_data_size, NULL, &error);
	if (error != CL_SUCCESS) {
		fprintf(stderr, "clCreateBuffer for OUT failed: %d\n", (int)error);
		free(a); free(b); free(out); free(out_normal);
		clReleaseMemObject(a_cl);
		clReleaseMemObject(b_cl);
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
		return 1;
	}

	srand(334);
	for (i = 0; i < size * size; i++) a[i] = rand() / (float)RAND_MAX;
	for (i = 0; i < size * size; i++) b[i] = rand() / (float)RAND_MAX;

	printf("start calculation of size %d\n", size);
	time_start = get_time();

#define CALL_AND_CHECK(name, func) \
	if ((error = func) != CL_SUCCESS) { \
		fprintf(stderr, name " failed: %d\n", (int)error); \
		free(a); free(b); free(out); free(out_normal); \
		clReleaseMemObject(a_cl); \
		clReleaseMemObject(b_cl); \
		clReleaseMemObject(out_cl); \
		clReleaseProgram(program); \
		clReleaseCommandQueue(queue); \
		return 1; \
	}

#define SET_KERNEL_ARG(index, value) \
	CALL_AND_CHECK("clSetKernelArg for " #value, \
		clSetKernelArg(kernel, index, sizeof(value), &value))
	SET_KERNEL_ARG(0, a_cl)
	SET_KERNEL_ARG(1, b_cl)
	SET_KERNEL_ARG(2, out_cl)
	SET_KERNEL_ARG(3, size)
#undef SET_KERNEL_ARG

	CALL_AND_CHECK("clEnqueueWriteBuffer",
		clEnqueueWriteBuffer(queue, a_cl, CL_TRUE, 0, matrix_data_size, a, 0, NULL, NULL))
	CALL_AND_CHECK("clEnqueueWriteBuffer",
		clEnqueueWriteBuffer(queue, b_cl, CL_TRUE, 0, matrix_data_size, b, 0, NULL, NULL))
	CALL_AND_CHECK("clEnqueueNDRangeKernel",
		clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size,
			0, NULL, NULL))
	CALL_AND_CHECK("clEnqueueReadBuffer",
		clEnqueueReadBuffer(queue, out_cl, CL_TRUE, 0, matrix_data_size, out, 0, NULL, NULL))
	CALL_AND_CHECK("clFinish", clFinish(queue))

	time_end = get_time();
	printf("OpenCL calculaton time: %f seconds (%fGflop/s)\n", time_end - time_start,
		flop / (time_end - time_start) * 1e-9);
	time_start = get_time();

	matmul_normal(a, b, out_normal, size);

	time_end = get_time();
	printf("normal calculaton time: %f seconds (%fGflop/s)\n", time_end - time_start,
		flop / (time_end - time_start) * 1e-9);

	diff_sum = 0;
	for (i = 0; i < size * size; i++) {
		diff_sum += fabs(out[i] - out_normal[i]);
	}
	printf("average difference: %g\n", diff_sum / (size * size));

#undef CALL_AND_CHECK
	free(a);
	free(b);
	free(out);
	free(out_normal);
	clReleaseMemObject(a_cl);
	clReleaseMemObject(b_cl);
	clReleaseMemObject(out_cl);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	return 0;
}
