#include <jni.h> 
#include <dlfcn.h>
#include <cstring>

#include "java.h"

const bool gDebug = false;

struct JavaData {
    JavaVM *jvm;       /* denotes a Java VM */
    JNIEnv *env;       /* pointer to native method interface */
    jobject obj;
    jclass cls;
};

class JavaImpl : public Java {
private:
    JavaVM *jvm;       /* denotes a Java VM */
    JNIEnv *env;       /* pointer to native method interface */
};

typedef 
jint JNICALL JNI_GetDefaultJavaVMInitArgs_tfunc(void *args);

typedef
jint JNICALL JNI_CreateJavaVM_tfunc(JavaVM **pvm, void **penv, void *args);

void *dr = nullptr;

// https://stackoverflow.com/questions/7506329/embed-java-into-a-c-application
bool Java::init() {
    if (this->data!=NULL) // already initialized?
        return true
        ;
    
    JavaData *data = new JavaData();
    if (dr==nullptr) {
        if (gDebug)
            printf("Loading java lib...\n");
        dr = dlopen("/usr/lib/jvm/java-17-openjdk-amd64/lib/server/libjvm.so", RTLD_LAZY);
    }
    if (dr==NULL)
        return false;

    JNI_GetDefaultJavaVMInitArgs_tfunc *JNI_GetDefaultJavaVMInitArgs = (JNI_GetDefaultJavaVMInitArgs_tfunc*)dlsym(dr,"JNI_GetDefaultJavaVMInitArgs");
    JNI_CreateJavaVM_tfunc *JNI_CreateJavaVM = (JNI_CreateJavaVM_tfunc*)dlsym(dr,"JNI_CreateJavaVM");

    JavaVMInitArgs vm_args; /* JDK 1.1 VM initialization arguments */
    
    vm_args.ignoreUnrecognized = 0;
    vm_args.nOptions = 1;
    vm_args.options = new JavaVMOption[1];
    vm_args.options[0].optionString = (char*)"-Djava.class.path=/home/user/numpy-cpp/environment/.class";
    vm_args.options[0].extraInfo = NULL;
    vm_args.version = JNI_VERSION_10; /* New in 1.1.2: VM version */
    /* Get the default initialization arguments and set the class 
     * path */
    jint r1 = JNI_GetDefaultJavaVMInitArgs(&vm_args);
    if (r1 != JNI_OK)
        return false;
    /* load and initialize a Java VM, return a JNI interface 
     * pointer in env */
    if (gDebug)
        printf("Creating vm...\n");
    jint r2 = JNI_CreateJavaVM(&data->jvm, (void**)&data->env, &vm_args);
    if (r2 != JNI_OK)
        return false;

    if (gDebug)
        printf("Loading class...\n");
    data->cls = data->env->FindClass("dev/damaso/Main");
    if (data->cls==nullptr) 
        return false;

    data->obj = data->env->AllocObject(data->cls);
    if (data->obj==nullptr)
        return false;

    this->data = (void*)data;
    return true;
}

void Java::copyImage(float *buffer) {
    JavaData *data = (JavaData*)this->data;
    jmethodID mid = data->env->GetMethodID(data->cls, "getImage", "()[[F");
    if (mid==nullptr) return;
    jobjectArray matrix = (jobjectArray)data->env->CallObjectMethod(data->obj, mid);
    if (matrix==nullptr) return;
    //return data->env->GetArrayLength(res);
    Dims dims = this->getImageDims();
    for (int i=0;i<dims.height;i++) {
        jfloatArray vector = (jfloatArray)data->env->GetObjectArrayElement(matrix, i);
        float *f = data->env->GetFloatArrayElements(vector, 0);
        for (int j=0;j<dims.width;j++) {
            buffer[i*dims.width+j] = f[j];
        }
        data->env->ReleaseFloatArrayElements(vector, f, 0);
    }
}

Java::Dims Java::getImageDims() {
    Dims dims;
    JavaData *data = (JavaData*)this->data;
    jmethodID mid = data->env->GetMethodID(data->cls, "getImage", "()[[F");
    if (mid==nullptr) return dims;
    jobjectArray matrix = (jobjectArray)data->env->CallObjectMethod(data->obj, mid);
    if (matrix==nullptr) return dims;
    dims.height = data->env->GetArrayLength(matrix);
    jobjectArray vector = (jobjectArray)data->env->GetObjectArrayElement(matrix, 0);
    dims.width = data->env->GetArrayLength(vector);
    return dims;
}
