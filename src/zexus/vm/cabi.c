#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "cabi.h"

// ---- C ABI functions (callable from native JIT) ----
PyObject *zexus_cabi_call_callable(PyObject *callable, PyObject *arg_tuple) {
    if (!callable || !PyCallable_Check(callable)) {
        Py_RETURN_NONE;
    }
    return PyObject_CallObject(callable, arg_tuple);
}

PyObject *zexus_cabi_call_method(PyObject *obj, PyObject *method_name, PyObject *arg_tuple) {
    if (!obj || !method_name) {
        Py_RETURN_NONE;
    }
    PyObject *method = PyObject_GetAttr(obj, method_name);
    if (!method) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    if (!PyCallable_Check(method)) {
        Py_DECREF(method);
        Py_RETURN_NONE;
    }
    PyObject *result = PyObject_CallObject(method, arg_tuple);
    Py_DECREF(method);
    return result;
}

PyObject *zexus_cabi_call_name(PyObject *env, PyObject *builtins, PyObject *name, PyObject *arg_tuple) {
    PyObject *callable = NULL;
    if (env && name) {
        callable = PyObject_GetItem(env, name);
        if (!callable) {
            PyErr_Clear();
        }
    }
    if (!callable && builtins && name) {
        callable = PyObject_GetItem(builtins, name);
        if (!callable) {
            PyErr_Clear();
        }
    }
    if (!callable || !PyCallable_Check(callable)) {
        Py_XDECREF(callable);
        Py_RETURN_NONE;
    }
    PyObject *result = PyObject_CallObject(callable, arg_tuple);
    Py_DECREF(callable);
    return result;
}

PyObject *zexus_cabi_build_list_from_array(PyObject **items, Py_ssize_t count) {
    PyObject *list = PyList_New(count);
    if (!list) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *item = items[i] ? items[i] : Py_None;
        Py_INCREF(item);
        PyList_SetItem(list, i, item);
    }
    return list;
}

PyObject *zexus_cabi_build_map_from_array(PyObject **items, Py_ssize_t count) {
    PyObject *dict = PyDict_New();
    if (!dict) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i + 1 < count; i += 2) {
        PyObject *k = items[i] ? items[i] : Py_None;
        PyObject *v = items[i + 1] ? items[i + 1] : Py_None;
        PyDict_SetItem(dict, k, v);
    }
    return dict;
}

PyObject *zexus_cabi_env_get(PyObject *env, PyObject *name) {
    if (!env || !name) {
        Py_RETURN_NONE;
    }
    PyObject *val = PyObject_GetItem(env, name);
    if (!val) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    return val;
}

int zexus_cabi_env_set(PyObject *env, PyObject *name, PyObject *value) {
    if (!env || !name) {
        return 0;
    }
    if (PyObject_SetItem(env, name, value) == 0) {
        return 1;
    }
    PyErr_Clear();
    return 0;
}

PyObject *zexus_cabi_number_add(PyObject *a, PyObject *b) { return PyNumber_Add(a, b); }
PyObject *zexus_cabi_number_sub(PyObject *a, PyObject *b) { return PyNumber_Subtract(a, b); }
PyObject *zexus_cabi_number_mul(PyObject *a, PyObject *b) { return PyNumber_Multiply(a, b); }
PyObject *zexus_cabi_number_div(PyObject *a, PyObject *b) { return PyNumber_TrueDivide(a, b); }
PyObject *zexus_cabi_number_mod(PyObject *a, PyObject *b) { return PyNumber_Remainder(a, b); }
PyObject *zexus_cabi_number_neg(PyObject *a) { return PyNumber_Negative(a); }

PyObject *zexus_cabi_compare_eq(PyObject *a, PyObject *b) {
    int r = PyObject_RichCompareBool(a, b, Py_EQ);
    if (r < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (r) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
PyObject *zexus_cabi_compare_ne(PyObject *a, PyObject *b) {
    int r = PyObject_RichCompareBool(a, b, Py_NE);
    if (r < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (r) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
PyObject *zexus_cabi_compare_lt(PyObject *a, PyObject *b) {
    int r = PyObject_RichCompareBool(a, b, Py_LT);
    if (r < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (r) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
PyObject *zexus_cabi_compare_gt(PyObject *a, PyObject *b) {
    int r = PyObject_RichCompareBool(a, b, Py_GT);
    if (r < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (r) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
PyObject *zexus_cabi_compare_lte(PyObject *a, PyObject *b) {
    int r = PyObject_RichCompareBool(a, b, Py_LE);
    if (r < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (r) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
PyObject *zexus_cabi_compare_gte(PyObject *a, PyObject *b) {
    int r = PyObject_RichCompareBool(a, b, Py_GE);
    if (r < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (r) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}

int zexus_cabi_truthy_int(PyObject *a) {
    int t = PyObject_IsTrue(a);
    if (t < 0) {
        PyErr_Clear();
        return 0;
    }
    return t ? 1 : 0;
}

PyObject *zexus_cabi_not(PyObject *a) {
    int t = PyObject_IsTrue(a);
    if (t < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (!t) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}

PyObject *zexus_cabi_bool_and(PyObject *a, PyObject *b) {
    int ta = zexus_cabi_truthy_int(a);
    int tb = zexus_cabi_truthy_int(b);
    if (ta && tb) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}

PyObject *zexus_cabi_bool_or(PyObject *a, PyObject *b) {
    int ta = zexus_cabi_truthy_int(a);
    int tb = zexus_cabi_truthy_int(b);
    if (ta || tb) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}

PyObject *zexus_cabi_index(PyObject *obj, PyObject *idx) {
    if (!obj || !idx) { Py_RETURN_NONE; }
    PyObject *val = PyObject_GetItem(obj, idx);
    if (!val) { PyErr_Clear(); Py_RETURN_NONE; }
    return val;
}

PyObject *zexus_cabi_get_attr(PyObject *obj, PyObject *attr) {
    if (!obj || !attr) { Py_RETURN_NONE; }
    PyObject *val = PyObject_GetAttr(obj, attr);
    if (!val) { PyErr_Clear(); Py_RETURN_NONE; }
    return val;
}

PyObject *zexus_cabi_get_length(PyObject *obj) {
    if (!obj) { Py_RETURN_NONE; }
    Py_ssize_t len = PyObject_Length(obj);
    if (len < 0) { PyErr_Clear(); Py_RETURN_NONE; }
    return PyLong_FromSsize_t(len);
}

PyObject *zexus_cabi_build_tuple_from_array(PyObject **items, Py_ssize_t count) {
    PyObject *tuple = PyTuple_New(count);
    if (!tuple) return NULL;
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *item = items[i] ? items[i] : Py_None;
        Py_INCREF(item);
        PyTuple_SetItem(tuple, i, item);
    }
    return tuple;
}

// ---- Python-callable wrappers (existing) ----
// Call a Python callable with args tuple
static PyObject *zexus_call_callable(PyObject *self, PyObject *args) {
    PyObject *callable = NULL;
    PyObject *arg_tuple = NULL;
    if (!PyArg_ParseTuple(args, "OO", &callable, &arg_tuple)) {
        return NULL;
    }
    return zexus_cabi_call_callable(callable, arg_tuple);
}

// Call obj.method(*args)
static PyObject *zexus_call_method(PyObject *self, PyObject *args) {
    PyObject *obj = NULL;
    PyObject *method_name = NULL;
    PyObject *arg_tuple = NULL;
    if (!PyArg_ParseTuple(args, "OOO", &obj, &method_name, &arg_tuple)) {
        return NULL;
    }
    return zexus_cabi_call_method(obj, method_name, arg_tuple);
}

// Build a list from a Python sequence
static PyObject *zexus_build_list(PyObject *self, PyObject *args) {
    PyObject *seq = NULL;
    if (!PyArg_ParseTuple(args, "O", &seq)) {
        return NULL;
    }
    PyObject *list = PySequence_List(seq);
    if (!list) {
        Py_RETURN_NONE;
    }
    return list;
}

// Build a list from C array of PyObject*
static PyObject *zexus_build_list_from_array(PyObject *self, PyObject *args) {
    PyObject *addr_obj = NULL;
    Py_ssize_t count = 0;
    if (!PyArg_ParseTuple(args, "On", &addr_obj, &count)) {
        return NULL;
    }
    PyObject **items = (PyObject **)PyLong_AsVoidPtr(addr_obj);
    if (!items) {
        Py_RETURN_NONE;
    }
    return zexus_cabi_build_list_from_array(items, count);
}

// Build a dict from a sequence of (k, v)
static PyObject *zexus_build_map(PyObject *self, PyObject *args) {
    PyObject *seq = NULL;
    if (!PyArg_ParseTuple(args, "O", &seq)) {
        return NULL;
    }
    PyObject *dict = PyDict_New();
    if (!dict) {
        return NULL;
    }
    PyObject *iter = PyObject_GetIter(seq);
    if (!iter) {
        Py_DECREF(dict);
        Py_RETURN_NONE;
    }
    PyObject *item;
    while ((item = PyIter_Next(iter))) {
        if (PyTuple_Check(item) && PyTuple_Size(item) == 2) {
            PyObject *k = PyTuple_GetItem(item, 0);
            PyObject *v = PyTuple_GetItem(item, 1);
            if (k && v) {
                PyDict_SetItem(dict, k, v);
            }
        }
        Py_DECREF(item);
    }
    Py_DECREF(iter);
    return dict;
}

// Build a dict from C array of PyObject* [k0,v0,k1,v1,...]
static PyObject *zexus_build_map_from_array(PyObject *self, PyObject *args) {
    PyObject *addr_obj = NULL;
    Py_ssize_t count = 0;
    if (!PyArg_ParseTuple(args, "On", &addr_obj, &count)) {
        return NULL;
    }
    PyObject **items = (PyObject **)PyLong_AsVoidPtr(addr_obj);
    if (!items) {
        Py_RETURN_NONE;
    }
    return zexus_cabi_build_map_from_array(items, count);
}

// env get/set helpers
static PyObject *zexus_env_get(PyObject *self, PyObject *args) {
    PyObject *env = NULL;
    PyObject *name = NULL;
    if (!PyArg_ParseTuple(args, "OO", &env, &name)) {
        return NULL;
    }
    return zexus_cabi_env_get(env, name);
}

static PyObject *zexus_env_set(PyObject *self, PyObject *args) {
    PyObject *env = NULL;
    PyObject *name = NULL;
    PyObject *value = NULL;
    if (!PyArg_ParseTuple(args, "OOO", &env, &name, &value)) {
        return NULL;
    }
    int ok = zexus_cabi_env_set(env, name, value);
    if (ok) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}

// Resolve and call a name from env/builtins
static PyObject *zexus_call_name(PyObject *self, PyObject *args) {
    PyObject *env = NULL;
    PyObject *builtins = NULL;
    PyObject *name = NULL;
    PyObject *arg_tuple = NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &env, &builtins, &name, &arg_tuple)) {
        return NULL;
    }
    return zexus_cabi_call_name(env, builtins, name, arg_tuple);
}

// Numeric ops via Python C API
static PyObject *zexus_number_add(PyObject *self, PyObject *args) {
    PyObject *a=NULL,*b=NULL; if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    return zexus_cabi_number_add(a, b);
}
static PyObject *zexus_number_sub(PyObject *self, PyObject *args) {
    PyObject *a=NULL,*b=NULL; if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    return zexus_cabi_number_sub(a, b);
}
static PyObject *zexus_number_mul(PyObject *self, PyObject *args) {
    PyObject *a=NULL,*b=NULL; if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    return zexus_cabi_number_mul(a, b);
}
static PyObject *zexus_number_div(PyObject *self, PyObject *args) {
    PyObject *a=NULL,*b=NULL; if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    return zexus_cabi_number_div(a, b);
}
static PyObject *zexus_number_mod(PyObject *self, PyObject *args) {
    PyObject *a=NULL,*b=NULL; if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    return zexus_cabi_number_mod(a, b);
}
static PyObject *zexus_number_neg(PyObject *self, PyObject *args) {
    PyObject *a=NULL; if (!PyArg_ParseTuple(args, "O", &a)) return NULL;
    return zexus_cabi_number_neg(a);
}
static PyObject *zexus_truthy(PyObject *self, PyObject *args) {
    PyObject *a=NULL; if (!PyArg_ParseTuple(args, "O", &a)) return NULL;
    int t = zexus_cabi_truthy_int(a);
    if (t) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
static PyObject *zexus_not(PyObject *self, PyObject *args) {
    PyObject *a=NULL; if (!PyArg_ParseTuple(args, "O", &a)) return NULL;
    return zexus_cabi_not(a);
}

// Comparisons
static PyObject *zexus_compare_eq(PyObject *self, PyObject *args) {
    PyObject *a=NULL,*b=NULL; if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    return zexus_cabi_compare_eq(a, b);
}
static PyObject *zexus_compare_ne(PyObject *self, PyObject *args) {
    PyObject *a=NULL,*b=NULL; if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    return zexus_cabi_compare_ne(a, b);
}
static PyObject *zexus_compare_lt(PyObject *self, PyObject *args) {
    PyObject *a=NULL,*b=NULL; if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    return zexus_cabi_compare_lt(a, b);
}
static PyObject *zexus_compare_gt(PyObject *self, PyObject *args) {
    PyObject *a=NULL,*b=NULL; if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    return zexus_cabi_compare_gt(a, b);
}
static PyObject *zexus_compare_lte(PyObject *self, PyObject *args) {
    PyObject *a=NULL,*b=NULL; if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    return zexus_cabi_compare_lte(a, b);
}
static PyObject *zexus_compare_gte(PyObject *self, PyObject *args) {
    PyObject *a=NULL,*b=NULL; if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    return zexus_cabi_compare_gte(a, b);
}

static PyObject *zexus_index(PyObject *self, PyObject *args) {
    PyObject *obj=NULL,*idx=NULL; if (!PyArg_ParseTuple(args, "OO", &obj, &idx)) return NULL;
    return zexus_cabi_index(obj, idx);
}

static PyObject *zexus_get_attr(PyObject *self, PyObject *args) {
    PyObject *obj=NULL,*attr=NULL; if (!PyArg_ParseTuple(args, "OO", &obj, &attr)) return NULL;
    return zexus_cabi_get_attr(obj, attr);
}

static PyObject *zexus_get_length(PyObject *self, PyObject *args) {
    PyObject *obj=NULL; if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
    return zexus_cabi_get_length(obj);
}

static PyObject *zexus_get_symbols(PyObject *self, PyObject *args) {
    PyObject *symbols = PyDict_New();
    if (!symbols) {
        return NULL;
    }
    PyDict_SetItemString(symbols, "zexus_call_callable", PyLong_FromVoidPtr((void *)&zexus_cabi_call_callable));
    PyDict_SetItemString(symbols, "zexus_call_name", PyLong_FromVoidPtr((void *)&zexus_cabi_call_name));
    PyDict_SetItemString(symbols, "zexus_call_method", PyLong_FromVoidPtr((void *)&zexus_cabi_call_method));
    PyDict_SetItemString(symbols, "zexus_build_list", PyLong_FromVoidPtr((void *)&zexus_cabi_build_list_from_array));
    PyDict_SetItemString(symbols, "zexus_build_map", PyLong_FromVoidPtr((void *)&zexus_cabi_build_map_from_array));
    PyDict_SetItemString(symbols, "zexus_build_list_from_array", PyLong_FromVoidPtr((void *)&zexus_cabi_build_list_from_array));
    PyDict_SetItemString(symbols, "zexus_build_map_from_array", PyLong_FromVoidPtr((void *)&zexus_cabi_build_map_from_array));
    PyDict_SetItemString(symbols, "zexus_env_get", PyLong_FromVoidPtr((void *)&zexus_cabi_env_get));
    PyDict_SetItemString(symbols, "zexus_env_set", PyLong_FromVoidPtr((void *)&zexus_cabi_env_set));
    PyDict_SetItemString(symbols, "zexus_number_add", PyLong_FromVoidPtr((void *)&zexus_cabi_number_add));
    PyDict_SetItemString(symbols, "zexus_number_sub", PyLong_FromVoidPtr((void *)&zexus_cabi_number_sub));
    PyDict_SetItemString(symbols, "zexus_number_mul", PyLong_FromVoidPtr((void *)&zexus_cabi_number_mul));
    PyDict_SetItemString(symbols, "zexus_number_div", PyLong_FromVoidPtr((void *)&zexus_cabi_number_div));
    PyDict_SetItemString(symbols, "zexus_number_mod", PyLong_FromVoidPtr((void *)&zexus_cabi_number_mod));
    PyDict_SetItemString(symbols, "zexus_number_neg", PyLong_FromVoidPtr((void *)&zexus_cabi_number_neg));
    PyDict_SetItemString(symbols, "zexus_truthy", PyLong_FromVoidPtr((void *)&zexus_cabi_truthy_int));
    PyDict_SetItemString(symbols, "zexus_truthy_int", PyLong_FromVoidPtr((void *)&zexus_cabi_truthy_int));
    PyDict_SetItemString(symbols, "zexus_not", PyLong_FromVoidPtr((void *)&zexus_cabi_not));
    PyDict_SetItemString(symbols, "zexus_bool_and", PyLong_FromVoidPtr((void *)&zexus_cabi_bool_and));
    PyDict_SetItemString(symbols, "zexus_bool_or", PyLong_FromVoidPtr((void *)&zexus_cabi_bool_or));
    PyDict_SetItemString(symbols, "zexus_compare_eq", PyLong_FromVoidPtr((void *)&zexus_cabi_compare_eq));
    PyDict_SetItemString(symbols, "zexus_compare_ne", PyLong_FromVoidPtr((void *)&zexus_cabi_compare_ne));
    PyDict_SetItemString(symbols, "zexus_compare_lt", PyLong_FromVoidPtr((void *)&zexus_cabi_compare_lt));
    PyDict_SetItemString(symbols, "zexus_compare_gt", PyLong_FromVoidPtr((void *)&zexus_cabi_compare_gt));
    PyDict_SetItemString(symbols, "zexus_compare_lte", PyLong_FromVoidPtr((void *)&zexus_cabi_compare_lte));
    PyDict_SetItemString(symbols, "zexus_compare_gte", PyLong_FromVoidPtr((void *)&zexus_cabi_compare_gte));
    PyDict_SetItemString(symbols, "zexus_index", PyLong_FromVoidPtr((void *)&zexus_cabi_index));
    PyDict_SetItemString(symbols, "zexus_get_attr", PyLong_FromVoidPtr((void *)&zexus_cabi_get_attr));
    PyDict_SetItemString(symbols, "zexus_get_length", PyLong_FromVoidPtr((void *)&zexus_cabi_get_length));
    PyDict_SetItemString(symbols, "zexus_build_tuple_from_array", PyLong_FromVoidPtr((void *)&zexus_cabi_build_tuple_from_array));
    return symbols;
}

static PyMethodDef ZexusCABIMethods[] = {
    {"call_callable", zexus_call_callable, METH_VARARGS, "Call a Python callable"},
    {"call_name", zexus_call_name, METH_VARARGS, "Resolve and call a name"},
    {"call_method", zexus_call_method, METH_VARARGS, "Call a method on a Python object"},
    {"build_list", zexus_build_list, METH_VARARGS, "Build list from sequence"},
    {"build_map", zexus_build_map, METH_VARARGS, "Build dict from sequence of pairs"},
    {"build_list_from_array", zexus_build_list_from_array, METH_VARARGS, "Build list from C array"},
    {"build_map_from_array", zexus_build_map_from_array, METH_VARARGS, "Build dict from C array"},
    {"env_get", zexus_env_get, METH_VARARGS, "Get item from env"},
    {"env_set", zexus_env_set, METH_VARARGS, "Set item in env"},
    {"number_add", zexus_number_add, METH_VARARGS, "Add numbers"},
    {"number_sub", zexus_number_sub, METH_VARARGS, "Subtract numbers"},
    {"number_mul", zexus_number_mul, METH_VARARGS, "Multiply numbers"},
    {"number_div", zexus_number_div, METH_VARARGS, "Divide numbers"},
    {"number_mod", zexus_number_mod, METH_VARARGS, "Modulo"},
    {"number_neg", zexus_number_neg, METH_VARARGS, "Negate"},
    {"truthy", zexus_truthy, METH_VARARGS, "Truthiness"},
    {"not", zexus_not, METH_VARARGS, "Not"},
    {"compare_eq", zexus_compare_eq, METH_VARARGS, "Compare EQ"},
    {"compare_ne", zexus_compare_ne, METH_VARARGS, "Compare NE"},
    {"compare_lt", zexus_compare_lt, METH_VARARGS, "Compare LT"},
    {"compare_gt", zexus_compare_gt, METH_VARARGS, "Compare GT"},
    {"compare_lte", zexus_compare_lte, METH_VARARGS, "Compare LTE"},
    {"compare_gte", zexus_compare_gte, METH_VARARGS, "Compare GTE"},
    {"index", zexus_index, METH_VARARGS, "Index"},
    {"get_attr", zexus_get_attr, METH_VARARGS, "Get attr"},
    {"get_length", zexus_get_length, METH_VARARGS, "Get length"},
    {"get_symbols", zexus_get_symbols, METH_VARARGS, "Return symbol addresses for native JIT"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef zexus_cabi_module = {
    PyModuleDef_HEAD_INIT,
    "cabi",
    "Zexus C ABI bridge",
    -1,
    ZexusCABIMethods
};

PyMODINIT_FUNC PyInit_cabi(void) {
    return PyModule_Create(&zexus_cabi_module);
}
