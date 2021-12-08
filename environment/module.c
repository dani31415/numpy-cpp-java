// https://docs.python.org/3/extending/extending.html
// https://numpy.org/doc/stable/reference/c-api/index.html
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/core/include/numpy/ndarrayobject.h>

static PyObject *SpamError;

static PyObject *
spam_system(PyObject *self, PyObject *args)
{
    const char *command;
    int sts;

    if (!PyArg_ParseTuple(args, "s", &command))
        return NULL;
    sts = system(command);
    return PyLong_FromLong(sts);
}

static PyObject *
spam_ndim(PyObject *self, PyObject *args)
{
    PyArrayObject *vec;
    // http://web.mit.edu/people/amliu/vrut/python/ext/parseTuple.html
    if (!PyArg_ParseTuple(args, "O", &vec))
        return NULL;
    
    if (!PyArray_Check(vec)) {
        PyErr_SetString(SpamError, "Expected numpy array.");
        return NULL;
    }

    int dims = PyArray_NDIM(vec);
    return PyLong_FromLong(dims);
}

static PyMethodDef SpamMethods[] = {
    {"system",  spam_system, METH_VARARGS,
     "Execute a shell command."},
    {"ndim",  spam_ndim, METH_VARARGS,
     "Get number of dimensions."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef spammodule = {
    PyModuleDef_HEAD_INIT,
    "spam",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    SpamMethods
};

PyMODINIT_FUNC
PyInit_spam(void)
{
    import_array(); // Initialize Numpy

    // Create module
    PyObject *m;    
    m = PyModule_Create(&spammodule);
    if (m == NULL)
        return NULL;

    // Create exception
    SpamError = PyErr_NewException("spam.error", NULL, NULL);
    Py_XINCREF(SpamError);
    if (PyModule_AddObject(m, "error", SpamError) < 0) {
        Py_XDECREF(SpamError);
        Py_CLEAR(SpamError);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}