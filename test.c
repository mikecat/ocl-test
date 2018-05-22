#include <stdio.h>
#include <stdlib.h>
#include "CL/cl.h"

int main(void) {
	cl_int error;
	cl_uint platformNum;
	cl_platform_id *platformIDs;
	cl_uint i;

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

	for (i = 0; i < platformNum; i++) {
		char name[4096];
		cl_uint deviceNum;
		cl_device_id *deviceIDs;
		size_t size;
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

	free(platformIDs);
	return 0;
}
