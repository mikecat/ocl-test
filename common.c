#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"

int calc_main(cl_platform_id, cl_device_id, cl_context, int, char**);

void CL_CALLBACK errorCallback(const char* errinfo,
const void* private_info, size_t cb, void* user_data) {
	fprintf(stderr, "OpenCL error: %s\n", errinfo);
}

int main(int argc, char* argv[]) {
	cl_int error;
	cl_uint platformNum;
	cl_platform_id *platformIDs;
	cl_uint deviceNum;
	cl_device_id *deviceIDs;
	int ret = 0;

	if ((error = clGetPlatformIDs(0, NULL, &platformNum)) != CL_SUCCESS) {
		fprintf(stderr, "get # of platform failed: %d\n", (int)error);
		return 1;
	}
	if ((platformIDs = malloc(sizeof(*platformIDs) * platformNum)) == NULL) {
		perror("malloc for platform failed");
		return 1;
	}
	if ((error = clGetPlatformIDs(platformNum, platformIDs, NULL)) != CL_SUCCESS) {
		fprintf(stderr, "get platform IDs failed: %d\n", (int)error);
		return 1;
	}

	if (argc < 3) {
		char name[4096];
		size_t size;
		cl_uint i, j;
		for (i = 0; i < platformNum; i++) {
			printf("platform %d: ", (int)i);
			if ((error = clGetPlatformInfo(platformIDs[i], CL_PLATFORM_NAME,
			sizeof(name), name, &size)) != CL_SUCCESS) {
				printf("(failed to get name: %d)\n", (int)error);
			} else {
				printf("%s\n", name);
			}

			if ((error = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL,
			0, NULL, &deviceNum)) != CL_SUCCESS) {
				printf("  (failed to get # of devices)\n");
			} else {
				deviceIDs = malloc(sizeof(*deviceIDs) * deviceNum);
				if (deviceIDs == NULL) {
					perror("failed to allocate deviceIDs");
				} else if ((error = clGetDeviceIDs(platformIDs[i], CL_DEVICE_TYPE_ALL,
				deviceNum, deviceIDs, NULL)) != CL_SUCCESS) {
					printf("  (failed to get device IDs\n");
				} else {
					cl_uint j;
					for (j = 0; j < deviceNum; j++) {
						printf("  device %d: ", (int)j);
						if ((error = clGetDeviceInfo(deviceIDs[j], CL_DEVICE_NAME,
						sizeof(name), name, &size)) != CL_SUCCESS) {
							printf("(failed to get name: %d)\n", (int)error);
						} else {
							printf("%s\n", name);
						}
					}
				}
				free(deviceIDs);
			}
		}
	} else {
		int platform = atoi(argv[1]);
		if (platform < 0 || platformNum <= (cl_uint)platform) {
			fprintf(stderr, "platform number out of range (only %u exists)\n",
				(unsigned int)platformNum);
			ret = 1;
		} else if ((error = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL,
		0, NULL, &deviceNum)) != CL_SUCCESS) {
			fputs("failed to get # of device IDs\n", stderr);
			ret = 1;
		} else {
			deviceIDs = malloc(sizeof(*deviceIDs) * deviceNum);
			if (deviceIDs == NULL) {
				perror("failed to allocate deviceIDs");
				ret = 1;
			} else if ((error = clGetDeviceIDs(platformIDs[platform], CL_DEVICE_TYPE_ALL,
			deviceNum, deviceIDs, NULL)) != CL_SUCCESS) {
				fputs("failed to get device IDs\n", stderr);
				ret = 1;
			} else {
				int device = atoi(argv[2]);
				if (device < 0 || deviceNum <= (cl_uint)device) {
					fprintf(stderr, "device number out of range (only %u exists)\n",
						(unsigned int )deviceNum);
					ret = 1;
				} else {
					cl_context_properties properties[3] = {CL_CONTEXT_PLATFORM, 0, 0};
					cl_context context;
					properties[1] = (cl_context_properties)platformIDs[platform];
					context = clCreateContext(properties, 1, &deviceIDs[device], errorCallback,
						NULL, &error);
					if (error != CL_SUCCESS) {
						fprintf(stderr, "clCreateContext failed: %d\n", (int)error);
						ret = 1;
					} else {
						ret = calc_main(platformIDs[platform], deviceIDs[device], context,
							argc - 3, argv + 3);
						clReleaseContext(context);
					}
				}
			}
			free(deviceIDs);
		}
	}

	free(platformIDs);
	return ret;
}

char* read_file(const char* fileName) {
	const size_t READ_CHUNK = 4096;
	FILE* fp;
	char* result = NULL;
	size_t start = 0;
	fp = fopen(fileName, "r");
	if (fp == NULL) return NULL;
	for (;;) {
		char* new_result = realloc(result, start + READ_CHUNK);
		size_t readLength;
		if (new_result == NULL) {
			free(result);
			return NULL;
		}
		readLength = fread(result + start, 1, READ_CHUNK, fp);
		if (readLength < READ_CHUNK) {
			if (feof(fp)) {
				result[start + readLength] = '\0';
				fclose(fp);
				return result;
			} else if (ferror(fp)) {
				free(result);
				fclose(fp);
				return NULL;
			}
		}
		start += readLength;
	}
}
