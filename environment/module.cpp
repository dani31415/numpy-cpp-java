// https://docs.python.org/3/extending/extending.html
// https://numpy.org/doc/stable/reference/c-api/index.html
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/core/include/numpy/ndarrayobject.h>

#include "model/country_side.h"

static PyObject *ModelError;

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
    CountrySide cs;
    matrix<float> & A = cs.getImage();
    npy_intp dims[2];
    dims[0] = A.m();;
    dims[1] = A.n();
    PyArrayObject * array = (PyArrayObject *)PyArray_SimpleNew(2, dims, NPY_FLOAT);
    float *buffer = (float*)PyArray_DATA(array);
    for (int i=0;i<dims[0];i++) {
        for (int j=0;j<dims[1];j++) {
            buffer[i*dims[1]+j] = A[i][j];
        }
    }
    // buffer[0] = 1.f;
    return (PyObject*)array;
}

static PyMethodDef modelMethods[] = {
    {"system",  model_system, METH_VARARGS,
     "Execute a shell command."},
    {"ndim",  model_ndim, METH_VARARGS,
     "Get number of dimensions."},
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
    import_array(); // Initialize Numpy

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

    return m;
}