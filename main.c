#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <CL/cl.h>

 // OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                       "\n" \
"__kernel void vecAdd(  __global double *a,                       \n" \
"                       __global double *b,                       \n" \
"                       __global double *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"}                                                               \n" \
                                                                "\n" ;

int main(int argc, char const *argv[])
{
   // Length of vectors
    unsigned int n = 100;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    // Host output vector
    double *h_c;

    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;

    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate memory for each vector on host
    h_a = (double*)malloc(bytes);
    h_b = (double*)malloc(bytes);
    h_c = (double*)malloc(bytes);

    // Initialize vectors on host
    int i;
    for( i = 0; i < n; i++ )
    {
        h_a[i] = 1;
        h_b[i] = 1;
    }

    /*
    Localsize refers to work-group (Where thread-items can perform sync, barreir operations and also have access to shared memory)
    */
    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group / work-group size
    localSize = 64;


    // Number of total work items - localSize must be devisor
    /*
    Even though our N == 100, and we only need 100 worker-threads for computation, but we still launch threads 
    which are multiple of localSize.
    */
    globalSize = ceil(n/(float)localSize)*localSize;
 
    // Bind to platform
    /*
    First Argument specifies how many platform handles to get
    Second Argument specifies the pointer where the platform handle information will be populated
    Third Argument is also a pointer, which can let us know how many platforms are actually present
    */
    err = clGetPlatformIDs(1, &cpPlatform, NULL);

    // cl_int num_platforms;
    // err = clGetPlatformIDs(1, NULL, &num_platforms);
    // printf("Number of Available Platforms = %d\n", num_platforms);

    // Get ID for the device
    /*
    First argument specifies, which platform to get devices against
    Second is the kind of the devices to get, in this case we are interested in CPU devices
    Third Argument tells us
    */
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
 
    // Create a context 
    /*
    An OpenCL context is created with one or more devices, using the clCreateContext()
    First argument is a properties argument
    The 2nd argument specifies the number of devices to use. The 3rd argument specifies the list of device handlers.
    4th is a callback function
    Last argument contains the error code, if there is any
    The context is used by the OpenCL runtime for managing objects, which will be discussed later.
    */
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    
    // Create a command queue
    /*
    The command queue is used to control the device.
    In OpenCL, any command from the host to the device, such as kernel execution or memory copy,
    is performed through this command queue. For each device, one or more command queue objects must be created. 

    The clCreateCommandQueue() function in line 51 creates the command queue.
    The 1st argument specifies the context in which the command queue will become part of.
    The 2nd argument specifies the device which will execute the command in the queue.
    The 3rd argument is the properties argument
    The function returns a handle to the command queue, which will be used for memory copy and kernel execution.
    */
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    

    // Create the compute program from the source buffer
    /*
    Once the source code is read, this code must be made into a kernel program.
    This step is required, since the kernel program can contain multiple kernel functions .
    This is done by creating a program object, as shown in the following code segment from hello.c

    The program object is created using the clCreateProgramWithSource() function. 
    The 3rd argument specifies the read-in source code, and the 4th argument specifies the size of the source code in bytes. 
    If the program object is to be created from a binary, clCreateProgramWithBinary() is used instead.
    */
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);
                            



    // Build the program executable
    /*
     First Argument is the program, which needs to be build
     Second argument is the number of target devices
     3rd Argument is the pointer to handles of target devices
     4th is options
     5th is a callback function
     6th is user data
    */
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vecAdd", &err);
 
    // Create the input and output arrays in device memory for our calculation
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
 
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, h_b, 0, NULL, NULL);








    // Set the arguments to our compute kernel
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
 
    // Execute the kernel over the entire range of the data set 
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    // Read the results from the device
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                bytes, h_c, 0, NULL, NULL );
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum);
 
    // release OpenCL resources
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    // //release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    return 0;
}
