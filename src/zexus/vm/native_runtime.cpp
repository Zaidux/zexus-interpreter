#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef PyObject* ZxValue;

extern "C" {

ZxValue zexus_rt_call_callable(ZxValue callable, ZxValue args_tuple) {
    if (!callable || !PyCallable_Check(callable)) {
        Py_RETURN_NONE;
    }
    return PyObject_CallObject(callable, args_tuple);
}

ZxValue zexus_rt_call_method(ZxValue obj, ZxValue method_name, ZxValue args_tuple) {
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
    PyObject *result = PyObject_CallObject(method, args_tuple);
    Py_DECREF(method);
    return result;
}

ZxValue zexus_rt_call_name(ZxValue env, ZxValue builtins, ZxValue name, ZxValue args_tuple) {
    PyObject *callable = NULL;
    if (env && name) {
        callable = PyObject_GetItem(env, name);
        if (!callable) { PyErr_Clear(); }
    }
    if (!callable && builtins && name) {
        callable = PyObject_GetItem(builtins, name);
        if (!callable) { PyErr_Clear(); }
    }
    if (!callable || !PyCallable_Check(callable)) {
        Py_XDECREF(callable);
        Py_RETURN_NONE;
    }
    PyObject *result = PyObject_CallObject(callable, args_tuple);
    Py_DECREF(callable);
    return result;
}

ZxValue zexus_rt_env_get(ZxValue env, ZxValue name) {
    if (!env || !name) {
        Py_RETURN_NONE;
    }
    PyObject *val = PyObject_GetItem(env, name);
    if (!val) { PyErr_Clear(); Py_RETURN_NONE; }
    return val;
}

int zexus_rt_env_set(ZxValue env, ZxValue name, ZxValue value) {
    if (!env || !name) return 0;
    if (PyObject_SetItem(env, name, value) == 0) return 1;
    PyErr_Clear();
    return 0;
}

ZxValue zexus_rt_build_list(ZxValue *items, Py_ssize_t count) {
    PyObject *list = PyList_New(count);
    if (!list) return NULL;
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *item = items[i] ? items[i] : Py_None;
        Py_INCREF(item);
        PyList_SetItem(list, i, item);
    }
    return list;
}

ZxValue zexus_rt_build_map(ZxValue *items, Py_ssize_t count) {
    PyObject *dict = PyDict_New();
    if (!dict) return NULL;
    for (Py_ssize_t i = 0; i + 1 < count; i += 2) {
        PyObject *k = items[i] ? items[i] : Py_None;
        PyObject *v = items[i + 1] ? items[i + 1] : Py_None;
        PyDict_SetItem(dict, k, v);
    }
    return dict;
}

ZxValue zexus_rt_build_tuple(ZxValue *items, Py_ssize_t count) {
    PyObject *tuple = PyTuple_New(count);
    if (!tuple) return NULL;
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *item = items[i] ? items[i] : Py_None;
        Py_INCREF(item);
        PyTuple_SetItem(tuple, i, item);
    }
    return tuple;
}

ZxValue zexus_rt_add(ZxValue a, ZxValue b) { return PyNumber_Add(a, b); }
ZxValue zexus_rt_sub(ZxValue a, ZxValue b) { return PyNumber_Subtract(a, b); }
ZxValue zexus_rt_mul(ZxValue a, ZxValue b) { return PyNumber_Multiply(a, b); }
ZxValue zexus_rt_div(ZxValue a, ZxValue b) { return PyNumber_TrueDivide(a, b); }
ZxValue zexus_rt_mod(ZxValue a, ZxValue b) { return PyNumber_Remainder(a, b); }
ZxValue zexus_rt_neg(ZxValue a) { return PyNumber_Negative(a); }

ZxValue zexus_rt_eq(ZxValue a, ZxValue b) {
    int r = PyObject_RichCompareBool(a, b, Py_EQ);
    if (r < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (r) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
ZxValue zexus_rt_neq(ZxValue a, ZxValue b) {
    int r = PyObject_RichCompareBool(a, b, Py_NE);
    if (r < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (r) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
ZxValue zexus_rt_lt(ZxValue a, ZxValue b) {
    int r = PyObject_RichCompareBool(a, b, Py_LT);
    if (r < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (r) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
ZxValue zexus_rt_gt(ZxValue a, ZxValue b) {
    int r = PyObject_RichCompareBool(a, b, Py_GT);
    if (r < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (r) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
ZxValue zexus_rt_lte(ZxValue a, ZxValue b) {
    int r = PyObject_RichCompareBool(a, b, Py_LE);
    if (r < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (r) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
ZxValue zexus_rt_gte(ZxValue a, ZxValue b) {
    int r = PyObject_RichCompareBool(a, b, Py_GE);
    if (r < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (r) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}

int zexus_rt_truthy(ZxValue a) {
    int t = PyObject_IsTrue(a);
    if (t < 0) { PyErr_Clear(); return 0; }
    return t ? 1 : 0;
}
ZxValue zexus_rt_not(ZxValue a) {
    int t = PyObject_IsTrue(a);
    if (t < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (!t) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
ZxValue zexus_rt_and(ZxValue a, ZxValue b) {
    int ta = zexus_rt_truthy(a);
    int tb = zexus_rt_truthy(b);
    if (ta && tb) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}
ZxValue zexus_rt_or(ZxValue a, ZxValue b) {
    int ta = zexus_rt_truthy(a);
    int tb = zexus_rt_truthy(b);
    if (ta || tb) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}

ZxValue zexus_rt_index(ZxValue obj, ZxValue idx) {
    if (!obj || !idx) { Py_RETURN_NONE; }
    PyObject *val = PyObject_GetItem(obj, idx);
    if (!val) { PyErr_Clear(); Py_RETURN_NONE; }
    return val;
}
ZxValue zexus_rt_get_attr(ZxValue obj, ZxValue attr) {
    if (!obj || !attr) { Py_RETURN_NONE; }
    PyObject *val = PyObject_GetAttr(obj, attr);
    if (!val) { PyErr_Clear(); Py_RETURN_NONE; }
    return val;
}
ZxValue zexus_rt_get_length(ZxValue obj) {
    if (!obj) { Py_RETURN_NONE; }
    Py_ssize_t len = PyObject_Length(obj);
    if (len < 0) { PyErr_Clear(); Py_RETURN_NONE; }
    return PyLong_FromSsize_t(len);
}

} // extern "C"

static PyObject *native_get_symbols(PyObject *self, PyObject *args) {
    PyObject *symbols = PyDict_New();
    if (!symbols) return NULL;
    PyDict_SetItemString(symbols, "zexus_call_callable", PyLong_FromVoidPtr((void *)&zexus_rt_call_callable));
    PyDict_SetItemString(symbols, "zexus_call_method", PyLong_FromVoidPtr((void *)&zexus_rt_call_method));
    PyDict_SetItemString(symbols, "zexus_call_name", PyLong_FromVoidPtr((void *)&zexus_rt_call_name));
    PyDict_SetItemString(symbols, "zexus_env_get", PyLong_FromVoidPtr((void *)&zexus_rt_env_get));
    PyDict_SetItemString(symbols, "zexus_env_set", PyLong_FromVoidPtr((void *)&zexus_rt_env_set));
    PyDict_SetItemString(symbols, "zexus_build_list", PyLong_FromVoidPtr((void *)&zexus_rt_build_list));
    PyDict_SetItemString(symbols, "zexus_build_map", PyLong_FromVoidPtr((void *)&zexus_rt_build_map));
    PyDict_SetItemString(symbols, "zexus_build_tuple_from_array", PyLong_FromVoidPtr((void *)&zexus_rt_build_tuple));
    PyDict_SetItemString(symbols, "zexus_number_add", PyLong_FromVoidPtr((void *)&zexus_rt_add));
    PyDict_SetItemString(symbols, "zexus_number_sub", PyLong_FromVoidPtr((void *)&zexus_rt_sub));
    PyDict_SetItemString(symbols, "zexus_number_mul", PyLong_FromVoidPtr((void *)&zexus_rt_mul));
    PyDict_SetItemString(symbols, "zexus_number_div", PyLong_FromVoidPtr((void *)&zexus_rt_div));
    PyDict_SetItemString(symbols, "zexus_number_mod", PyLong_FromVoidPtr((void *)&zexus_rt_mod));
    PyDict_SetItemString(symbols, "zexus_number_neg", PyLong_FromVoidPtr((void *)&zexus_rt_neg));
    PyDict_SetItemString(symbols, "zexus_compare_eq", PyLong_FromVoidPtr((void *)&zexus_rt_eq));
    PyDict_SetItemString(symbols, "zexus_compare_ne", PyLong_FromVoidPtr((void *)&zexus_rt_neq));
    PyDict_SetItemString(symbols, "zexus_compare_lt", PyLong_FromVoidPtr((void *)&zexus_rt_lt));
    PyDict_SetItemString(symbols, "zexus_compare_gt", PyLong_FromVoidPtr((void *)&zexus_rt_gt));
    PyDict_SetItemString(symbols, "zexus_compare_lte", PyLong_FromVoidPtr((void *)&zexus_rt_lte));
    PyDict_SetItemString(symbols, "zexus_compare_gte", PyLong_FromVoidPtr((void *)&zexus_rt_gte));
    PyDict_SetItemString(symbols, "zexus_truthy_int", PyLong_FromVoidPtr((void *)&zexus_rt_truthy));
    PyDict_SetItemString(symbols, "zexus_not", PyLong_FromVoidPtr((void *)&zexus_rt_not));
    PyDict_SetItemString(symbols, "zexus_bool_and", PyLong_FromVoidPtr((void *)&zexus_rt_and));
    PyDict_SetItemString(symbols, "zexus_bool_or", PyLong_FromVoidPtr((void *)&zexus_rt_or));
    PyDict_SetItemString(symbols, "zexus_index", PyLong_FromVoidPtr((void *)&zexus_rt_index));
    PyDict_SetItemString(symbols, "zexus_get_attr", PyLong_FromVoidPtr((void *)&zexus_rt_get_attr));
    PyDict_SetItemString(symbols, "zexus_get_length", PyLong_FromVoidPtr((void *)&zexus_rt_get_length));
    return symbols;
}

static PyMethodDef NativeRuntimeMethods[] = {
    {"get_symbols", native_get_symbols, METH_VARARGS, "Return symbol addresses for native runtime"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef native_runtime_module = {
    PyModuleDef_HEAD_INIT,
    "native_runtime",
    "Zexus native runtime (C++)",
    -1,
    NativeRuntimeMethods
};

PyMODINIT_FUNC PyInit_native_runtime(void) {
    return PyModule_Create(&native_runtime_module);
}
