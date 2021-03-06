#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <immintrin.h>
#include "CL/cl.h"

char* read_file(const char* fileName);

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
	int i;
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < size; i++) {
		int j, k;
		for (j = 0; j < size; j++) {
			float ret = 0;
			for (k = 0; k < size; k++) {
				ret += a[i * size + k] * b[k * size + j];
			}
			out[i * size + j] = ret;
		}
	}
}


void matmul_normal2(const float* a, const float* b, float* out, int size) {
	int i;
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < size; i++) {
		int j;
		for (j = 0; j < size; j++) {
			out[i * size + j] = 0;
		}
	}
	#ifdef _OPENMP
	#pragma omp parallel for
	#endif
	for (i = 0; i < size; i++) {
		int k, j;
		for (k = 0; k < size; k++) {
			for (j = 0; j < size; j++) {
				out[i * size + j] += a[i * size + k] * b[k * size + j];
			}
		}
	}
}

#ifndef CHUNK_I
#define CHUNK_I 128
#endif
#ifndef CHUNK_J
#define CHUNK_J 128
#endif
#ifndef CHUNK_K
#define CHUNK_K 128
#endif

void matmul_normal3(const float* a, const float* b, float* out, int size) {
	int i;
	#ifdef _OPENMP
	#pragma omp parallel
	#endif
	{
		float *a_buf, *b_buf, *out_buf;
		a_buf = malloc(sizeof(float) * CHUNK_I * CHUNK_K);
		b_buf = malloc(sizeof(float) * CHUNK_K * CHUNK_J);
		out_buf = malloc(sizeof(float) * CHUNK_I * CHUNK_J);
		if (a_buf == NULL || b_buf == NULL || out_buf == NULL) {
			fputs("malloc failed in matmul_normal3!\n", stderr);
			exit(1);
		}
		#ifdef _OPENMP
		#pragma omp for
		#endif
		for (i = 0; i < size; i += CHUNK_I) {
			int j, k, bi, bj, bk;
			int size_i = i + CHUNK_I <= size ? CHUNK_I : size - i;
			for (j = 0; j < size; j += CHUNK_J) {
				int size_j = j + CHUNK_J <= size ? CHUNK_J : size - j;
				for (bi = 0; bi < size_i; bi++) {
					for (bj = 0; bj < size_j; bj++) {
						out_buf[bi * size_j + bj] = 0;
					}
				}
				for (k = 0; k < size; k += CHUNK_K) {
					int size_k = k + CHUNK_K <= size ? CHUNK_K : size - k;
					for (bi = 0; bi < size_i; bi++) {
						memcpy(a_buf + bi * size_k, a + (i + bi) * size + k, sizeof(float) * size_k);
					}
					for (bk = 0; bk < size_k; bk++) {
						memcpy(b_buf + bk * size_j, b + (k + bk) * size + j, sizeof(float) * size_j);
					}
					for (bi = 0; bi < size_i; bi++) {
						for (bk = 0; bk < size_k; bk++) {
							for (bj = 0; bj < size_j; bj++) {
								out_buf[bi * size_j + bj] += a_buf[bi * size_k + bk] * b_buf[bk * size_j + bj];
							}
						}
					}
				}
				for (bi = 0; bi < size_i; bi++) {
					memcpy(out + (i + bi) * size + j, out_buf + bi * size_j, sizeof(float) * size_j);
				}
			}
		}
		free(a_buf);
		free(b_buf);
		free(out_buf);
	}
}

#if CHUNK_J % 8 != 0
#error CHUNK_J must be multiple of 8
#endif

void matmul_normal4(const float* a, const float* b, float* out, int size) {
	int i;
	#ifdef _OPENMP
	#pragma omp parallel
	#endif
	{
		float *a_buf_raw, *b_buf_raw, *out_buf_raw;
		float *a_buf, *b_buf, *out_buf;
		a_buf_raw = malloc(sizeof(float) * (CHUNK_I * CHUNK_K + 8));
		b_buf_raw = malloc(sizeof(float) * (CHUNK_K * CHUNK_J + 8));
		out_buf_raw = malloc(sizeof(float) * (CHUNK_I * CHUNK_J + 8));
		if (a_buf_raw == NULL || b_buf_raw == NULL || out_buf_raw == NULL) {
			fputs("malloc failed in matmul_normal3!\n", stderr);
			exit(1);
		}
		for (a_buf = a_buf_raw; (uintptr_t)a_buf % 32 != 0; a_buf++);
		for (b_buf = b_buf_raw; (uintptr_t)b_buf % 32 != 0; b_buf++);
		for (out_buf = out_buf_raw; (uintptr_t)out_buf % 32 != 0; out_buf++);
		#ifdef _OPENMP
		#pragma omp for
		#endif
		for (i = 0; i < size; i += CHUNK_I) {
			int j, k, bi, bj, bk;
			int size_i = i + CHUNK_I <= size ? CHUNK_I : size - i;
			for (j = 0; j < size; j += CHUNK_J) {
				int size_j = j + CHUNK_J <= size ? CHUNK_J : size - j;
				__m256 zero = _mm256_setzero_ps();
				for (bi = 0; bi < size_i; bi++) {
					bj = 0;
					for (; bj + 8 <= size_j; bj += 8) {
						_mm256_store_ps(&out_buf[bi * CHUNK_J + bj], zero);
					}
					for (; bj < size_j; bj++) {
						out_buf[bi * CHUNK_J + bj] = 0;
					}
				}
				for (k = 0; k < size; k += CHUNK_K) {
					int size_k = k + CHUNK_K <= size ? CHUNK_K : size - k;
					for (bi = 0; bi < size_i; bi++) {
						memcpy(a_buf + bi * CHUNK_K, a + (i + bi) * size + k, sizeof(float) * size_k);
					}
					for (bk = 0; bk < size_k; bk++) {
						memcpy(b_buf + bk * CHUNK_J, b + (k + bk) * size + j, sizeof(float) * size_j);
					}
					for (bi = 0; bi < size_i; bi++) {
						for (bk = 0; bk < size_k; bk++) {
							__m256 a_mm = _mm256_set1_ps(a_buf[bi * CHUNK_K + bk]);
							bj = 0;
							for (; bj + 8 <= size_j; bj += 8) {
								__m256 out_mm, b_mm;
								out_mm = _mm256_load_ps(&out_buf[bi * CHUNK_J + bj]);
								b_mm = _mm256_load_ps(&b_buf[bk * CHUNK_J + bj]);
								out_mm = _mm256_fmadd_ps(a_mm, b_mm, out_mm);
								_mm256_store_ps(&out_buf[bi * CHUNK_J + bj], out_mm);
							}
							for (; bj < size_j; bj++) {
								out_buf[bi * CHUNK_J + bj] += a_buf[bi * CHUNK_K + bk] * b_buf[bk * CHUNK_J + bj];
							}
						}
					}
				}
				for (bi = 0; bi < size_i; bi++) {
					memcpy(out + (i + bi) * size + j, out_buf + bi * CHUNK_J, sizeof(float) * size_j);
				}
			}
		}
		free(a_buf_raw);
		free(b_buf_raw);
		free(out_buf_raw);
	}
}

int calc_main(cl_platform_id platform, cl_device_id device, cl_context context,
int argc, char* argv[]) {
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_int error;

	int size = (argc > 0 ? atoi(argv[0]) : 1024);
	char* read_kernel = NULL;
	float *a, *b, *out, *out_normal;
	size_t matrix_data_size;
	cl_mem a_cl, b_cl, out_cl;
	size_t global_size[2], local_size[2] = {8, 8};
	char option[1024];
	double time_start, time_end, flop;
	int i;
	float diff_sum;
	int skip_normal = 0, skip_slow_normal = 0, skip_mid_normal = 0, skip_opencl = 0;
	char *env_str;

	env_str = getenv("SKIP_NORMAL");
	if (env_str != NULL) skip_normal = atoi(env_str);
	env_str = getenv("SKIP_SLOW_NORMAL");
	if (env_str != NULL) skip_slow_normal = atoi(env_str);
	env_str = getenv("SKIP_MID_NORMAL");
	if (env_str != NULL) skip_mid_normal = atoi(env_str);
	env_str = getenv("SKIP_OPENCL");
	if (env_str != NULL) skip_opencl= atoi(env_str);

	if (size <= 0) {
		fputs("matrix size must be positive\n", stderr);
		return 1;
	}

	if (argc > 1) {
		read_kernel = read_file(argv[1]);
		if (read_kernel != NULL) {
			printf("read kernel from %s\n", argv[1]);
		} else {
			printf("failed to read kernel from %s!\n", argv[1]);
		}
	}
	if (argc > 2) {
		int t_size;
		t_size = atoi(argv[2]);
		if (t_size <= 0) {
			fputs("local size must be positive\n", stderr);
			return 1;
		}
		local_size[0] = local_size[1] = t_size;
	}
	printf("local block size: %zu\n", local_size[0]);

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

	if (!skip_opencl) {
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

		program = clCreateProgramWithSource(context, 1,
			read_kernel != NULL ? (const char**)&read_kernel : &kernel_code, NULL, &error);
		if (error != CL_SUCCESS) {
			fprintf(stderr, "clCreateProgramWithSource failed: %d\n", (int)error);
			free(a); free(b); free(out); free(out_normal);
			clReleaseCommandQueue(queue);
			return 1;
		}

		snprintf(option, sizeof(option), "-w -D LOCAL_BLOCK_SIZE=%zu", local_size[0]);
		if ((error = clBuildProgram(program, 1, &device, option, NULL, NULL)) != CL_SUCCESS) {
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
	}

	srand(334);
	for (i = 0; i < size * size; i++) a[i] = rand() / (float)RAND_MAX;
	for (i = 0; i < size * size; i++) b[i] = rand() / (float)RAND_MAX;

	printf("start calculation of size %d\n", size);

	if (!skip_normal && !skip_slow_normal) {
		time_start = get_time();
		matmul_normal(a, b, out_normal, size);
		time_end = get_time();
		printf("normal  calculaton time: %f seconds (%fGflop/s)\n", time_end - time_start,
			flop / (time_end - time_start) * 1e-9);
	}

	if (!skip_normal) {
		if (!skip_mid_normal) {
			time_start = get_time();
			matmul_normal2(a, b, out, size);
			time_end = get_time();
			printf("normal2 calculaton time: %f seconds (%fGflop/s)\n", time_end - time_start,
				flop / (time_end - time_start) * 1e-9);
			if (!skip_slow_normal) {
				diff_sum = 0;
				for (i = 0; i < size * size; i++) {
					diff_sum += fabs(out[i] - out_normal[i]);
				}
				printf("average difference with normal: %g\n", diff_sum / (size * size));
			}

			time_start = get_time();
			matmul_normal3(a, b, out, size);
			time_end = get_time();
			printf("normal3 calculaton time: %f seconds (%fGflop/s)\n", time_end - time_start,
				flop / (time_end - time_start) * 1e-9);
			if (!skip_slow_normal) {
				diff_sum = 0;
				for (i = 0; i < size * size; i++) {
					diff_sum += fabs(out[i] - out_normal[i]);
				}
				printf("average difference with normal: %g\n", diff_sum / (size * size));
			}
		}

		time_start = get_time();
		matmul_normal4(a, b, out, size);
		time_end = get_time();
		printf("normal4 calculaton time: %f seconds (%fGflop/s)\n", time_end - time_start,
			flop / (time_end - time_start) * 1e-9);
		if (!skip_slow_normal) {
			diff_sum = 0;
			for (i = 0; i < size * size; i++) {
				diff_sum += fabs(out[i] - out_normal[i]);
			}
			printf("average difference with normal: %g\n", diff_sum / (size * size));
		}
	}

	if (!skip_opencl) {
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
		printf("OpenCL  calculaton time: %f seconds (%fGflop/s)\n", time_end - time_start,
			flop / (time_end - time_start) * 1e-9);
		if (!skip_normal && !skip_slow_normal) {
			diff_sum = 0;
			for (i = 0; i < size * size; i++) {
				diff_sum += fabs(out[i] - out_normal[i]);
			}
			printf("average difference with normal: %g\n", diff_sum / (size * size));
		}
	}

#undef CALL_AND_CHECK
	free(a);
	free(b);
	free(out);
	free(out_normal);
	free(read_kernel);
	if (!skip_opencl) {
		clReleaseMemObject(a_cl);
		clReleaseMemObject(b_cl);
		clReleaseMemObject(out_cl);
		clReleaseKernel(kernel);
		clReleaseProgram(program);
		clReleaseCommandQueue(queue);
	}
	return 0;
}
