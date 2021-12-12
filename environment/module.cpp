// https://docs.python.org/3/extending/extending.html
// https://numpy.org/doc/stable/reference/c-api/index.html
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/core/include/numpy/ndarrayobject.h>

#include "java.h"

static PyObject *ModelError;
static Java java;

static PyObject *
model_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyObject *
model_ndim(PyObject *self, PyObject *args)
{
    PyArrayObject *vec;
    // http://web.mit.edu/people/amliu/vrut/python/ext/parseTuple.html
    if (!PyArg_ParseTuple(args, "O", &vec))
        return NULL;
    
    if (!PyArray_Check(vec)) {
        PyErr_SetString(ModelError, "Expected numpy array.");
        return NULL;
    }

    int dims = PyArray_NDIM(vec);
    return PyLong_FromLong(dims);
}

static PyObject *
model_image(PyObject *self, PyObject *args)
{
    Java::Dims imageDims = java.getImageDims();
    npy_intp dims[2];
    dims[0] = imageDims.height;
    dims[1] = imageDims.width;
    PyArrayObject * array = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT);
    float *buffer = (float*)PyArray_DATA(array);
    java.copyImage(buffer);
    return (PyObject*)array;
}

static PyObject *
model_imageDims(PyObject *self, PyObject *args)
{
    Java::Dims imageDims = java.getImageDims();
    npy_intp dims[1];
    dims[0] = 2;
    PyArrayObject * array = (PyArrayObject *)PyArray_SimpleNew(1, dims, NPY_INT64);
    long *buffer = (long*)PyArray_DATA(array);
    buffer[0] = imageDims.height;
    buffer[1] = imageDims.width;
    return (PyObject*)array;
}

static PyMethodDef modelMethods[] = {
    {"system",  model_system, METH_VARARGS,
     "Execute a shell command."},
    {"ndim",  model_ndim, METH_VARARGS,
     "Get number of dimensions."},
    {"imageDims",  model_imageDims, METH_VARARGS,
     "Get dimensions of image."},
    {"image",  model_image, METH_VARARGS,
     "Get image."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef modelModule = {
    PyModuleDef_HEAD_INIT,
    "model",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    modelMethods
};

PyMODINIT_FUNC
PyInit_model(void)
{
    printf("Start initialized.\n");
    // Initialize Numpy
    import_array();

    // Initialize Java
    if (!java.init()) 
        return NULL;

    // Create module
    PyObject *m;    
    m = PyModule_Create(&modelModule);
    if (m == NULL)
        return NULL;

    // Create exception
    ModelError = PyErr_NewException("model.error", NULL, NULL);
    Py_XINCREF(ModelError);
    if (PyModule_AddObject(m, "error", ModelError) < 0) {
        Py_XDECREF(ModelError);
        Py_CLEAR(ModelError);
        Py_DECREF(m);
        return NULL;
    }
    printf("Module initialized.\n");
    return m;
}