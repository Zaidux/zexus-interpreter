#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "cabi.h"

static PyObject *zx_get_attr(PyObject *obj, const char *name) {
    if (!obj) return NULL;
    return PyObject_GetAttrString(obj, name);
}

static int zx_is_awaitable(PyObject *obj) {
    if (!obj) return 0;
    int has = PyObject_HasAttrString(obj, "__await__");
    return has > 0;
}

static PyObject *zx_to_bytes(PyObject *obj) {
    if (!obj) {
        return PyBytes_FromString("");
    }
    if (PyBytes_Check(obj)) {
        Py_INCREF(obj);
        return obj;
    }
    if (PyUnicode_Check(obj)) {
        return PyUnicode_AsEncodedString(obj, "utf-8", "strict");
    }
    if (PyDict_Check(obj)) {
        PyObject *json = PyImport_ImportModule("json");
        if (!json) { PyErr_Clear(); goto fallback; }
        PyObject *dumps = PyObject_GetAttrString(json, "dumps");
        Py_DECREF(json);
        if (!dumps) { PyErr_Clear(); goto fallback; }
        PyObject *kwargs = Py_BuildValue("{s:O}", "sort_keys", Py_True);
        PyObject *args = PyTuple_Pack(1, obj);
        PyObject *s = PyObject_Call(dumps, args, kwargs);
        Py_DECREF(args);
        Py_DECREF(kwargs);
        Py_DECREF(dumps);
        if (!s) { PyErr_Clear(); goto fallback; }
        PyObject *bytes = PyUnicode_AsEncodedString(s, "utf-8", "strict");
        Py_DECREF(s);
        return bytes;
    }
fallback:
    {
        PyObject *s = PyObject_Str(obj);
        if (!s) { PyErr_Clear(); return PyBytes_FromString(""); }
        PyObject *bytes = PyUnicode_AsEncodedString(s, "utf-8", "strict");
        Py_DECREF(s);
        return bytes;
    }
}

static PyObject *zx_sha256_hex(PyObject *bytes_obj) {
    PyObject *hashlib = PyImport_ImportModule("hashlib");
    if (!hashlib) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *sha256 = PyObject_GetAttrString(hashlib, "sha256");
    Py_DECREF(hashlib);
    if (!sha256) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *empty = NULL;
    PyObject *arg_obj = bytes_obj ? bytes_obj : (empty = PyBytes_FromString(""));
    PyObject *args = PyTuple_Pack(1, arg_obj);
    Py_XDECREF(empty);
    PyObject *hash = PyObject_CallObject(sha256, args);
    Py_DECREF(args);
    Py_DECREF(sha256);
    if (!hash) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *hex = PyObject_CallMethod(hash, "hexdigest", NULL);
    Py_DECREF(hash);
    if (!hex) { PyErr_Clear(); Py_RETURN_NONE; }
    return hex;
}

static PyObject *zx_env_get_dict(PyObject *env, const char *key, int create) {
    if (!env || !PyDict_Check(env)) return NULL;
    PyObject *k = PyUnicode_FromString(key);
    if (!k) return NULL;
    PyObject *val = PyDict_GetItem(env, k);
    if (!val && create) {
        PyObject *d = PyDict_New();
        if (d) {
            PyDict_SetItem(env, k, d);
            val = d;
            Py_DECREF(d);
        }
    }
    Py_DECREF(k);
    return val;
}

static PyObject *zx_env_get_bool(PyObject *env, const char *key) {
    if (!env || !PyDict_Check(env)) return NULL;
    PyObject *k = PyUnicode_FromString(key);
    if (!k) return NULL;
    PyObject *val = PyDict_GetItem(env, k);
    Py_DECREF(k);
    return val;
}

static void zx_env_set(PyObject *env, const char *key, PyObject *value) {
    if (!env || !PyDict_Check(env)) return;
    PyObject *k = PyUnicode_FromString(key);
    if (!k) return;
    PyDict_SetItem(env, k, value ? value : Py_None);
    Py_DECREF(k);
}

static PyObject *zx_env_get_list(PyObject *env, const char *key, int create) {
    if (!env || !PyDict_Check(env)) return NULL;
    PyObject *k = PyUnicode_FromString(key);
    if (!k) return NULL;
    PyObject *val = PyDict_GetItem(env, k);
    if (!val && create) {
        PyObject *lst = PyList_New(0);
        if (lst) {
            PyDict_SetItem(env, k, lst);
            val = lst;
            Py_DECREF(lst);
        }
    }
    Py_DECREF(k);
    return val;
}

static PyObject *zx_unwrap_value(PyObject *obj) {
    if (!obj) {
        Py_RETURN_NONE;
    }
    PyObject *attr = PyObject_GetAttrString(obj, "value");
    if (attr) {
        if (!PyCallable_Check(attr)) {
            return attr;
        }
        Py_DECREF(attr);
    } else {
        PyErr_Clear();
    }
    Py_INCREF(obj);
    return obj;
}

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

PyObject *zexus_cabi_build_set_from_array(PyObject **items, Py_ssize_t count) {
    PyObject *set = PySet_New(NULL);
    if (!set) {
        return NULL;
    }
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *item = items[i] ? items[i] : Py_None;
        PySet_Add(set, item);
    }
    return set;
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

PyObject *zexus_cabi_export(PyObject *env, PyObject *name, PyObject *value) {
    if (!env || !name) { Py_RETURN_NONE; }
    PyObject *val = value;
    if (!val || val == Py_None) {
        val = PyObject_GetItem(env, name);
        if (!val) {
            PyErr_Clear();
            val = Py_None;
            Py_INCREF(val);
        }
    } else {
        Py_INCREF(val);
    }

    PyObject *export_fn = PyObject_GetAttrString(env, "export");
    if (export_fn && PyCallable_Check(export_fn)) {
        PyObject *args = PyTuple_Pack(2, name, val);
        if (args) {
            PyObject *res = PyObject_CallObject(export_fn, args);
            Py_XDECREF(res);
            Py_DECREF(args);
        }
        Py_DECREF(export_fn);
    } else {
        Py_XDECREF(export_fn);
        if (PyObject_SetItem(env, name, val) != 0) {
            PyErr_Clear();
        }
    }

    Py_DECREF(val);
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_number_add(PyObject *a, PyObject *b) { return PyNumber_Add(a, b); }
PyObject *zexus_cabi_number_sub(PyObject *a, PyObject *b) { return PyNumber_Subtract(a, b); }
PyObject *zexus_cabi_number_mul(PyObject *a, PyObject *b) { return PyNumber_Multiply(a, b); }
PyObject *zexus_cabi_number_div(PyObject *a, PyObject *b) { return PyNumber_TrueDivide(a, b); }
PyObject *zexus_cabi_number_mod(PyObject *a, PyObject *b) { return PyNumber_Remainder(a, b); }
PyObject *zexus_cabi_number_pow(PyObject *a, PyObject *b) { return PyNumber_Power(a, b, Py_None); }
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

PyObject *zexus_cabi_slice(PyObject *obj, PyObject *start, PyObject *end) {
    if (!obj) { Py_RETURN_NONE; }
    PyObject *slice = PySlice_New(start ? start : Py_None, end ? end : Py_None, Py_None);
    if (!slice) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *res = PyObject_GetItem(obj, slice);
    Py_DECREF(slice);
    if (res) {
        return res;
    }
    PyErr_Clear();

    PyObject *elements = PyObject_GetAttrString(obj, "elements");
    if (!elements) {
        PyErr_Clear();
        Py_RETURN_NONE;
    }
    Py_ssize_t s = 0;
    Py_ssize_t e = PyList_Check(elements) ? PyList_Size(elements) : PySequence_Size(elements);
    if (start && start != Py_None) {
        s = PyLong_AsSsize_t(start);
    }
    if (end && end != Py_None) {
        e = PyLong_AsSsize_t(end);
    }
    if (PyErr_Occurred()) {
        PyErr_Clear();
        s = 0;
        e = PySequence_Size(elements);
    }
    PyObject *slice_list = PySequence_GetSlice(elements, s, e);
    Py_DECREF(elements);
    if (!slice_list) { PyErr_Clear(); Py_RETURN_NONE; }
    return slice_list;
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

PyObject *zexus_cabi_print(PyObject *obj) {
    if (!obj) {
        PySys_WriteStdout("%s", "null");
        PySys_WriteStdout("\n");
        Py_RETURN_NONE;
    }
    PyObject *str = PyObject_Str(obj);
    if (str) {
        const char *cstr = PyUnicode_AsUTF8(str);
        if (cstr) {
            PySys_WriteStdout("%s", cstr);
        }
        Py_DECREF(str);
    }
    PySys_WriteStdout("\n");
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_read(PyObject *path) {
    if (!path) { Py_RETURN_NONE; }
    PyObject *io = PyImport_ImportModule("io");
    if (!io) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *open_fn = PyObject_GetAttrString(io, "open");
    Py_DECREF(io);
    if (!open_fn) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *args = PyTuple_Pack(2, path, PyUnicode_FromString("r"));
    PyObject *file = PyObject_CallObject(open_fn, args);
    Py_DECREF(args);
    Py_DECREF(open_fn);
    if (!file) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *read_fn = PyObject_GetAttrString(file, "read");
    if (!read_fn) { PyErr_Clear(); Py_DECREF(file); Py_RETURN_NONE; }
    PyObject *content = PyObject_CallObject(read_fn, NULL);
    Py_DECREF(read_fn);
    PyObject *close_fn = PyObject_GetAttrString(file, "close");
    if (close_fn) {
        PyObject_CallObject(close_fn, NULL);
        Py_DECREF(close_fn);
    }
    Py_DECREF(file);
    if (!content) { PyErr_Clear(); Py_RETURN_NONE; }
    return content;
}

PyObject *zexus_cabi_write(PyObject *path, PyObject *content) {
    if (!path) { Py_RETURN_NONE; }
    PyObject *io = PyImport_ImportModule("io");
    if (!io) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *open_fn = PyObject_GetAttrString(io, "open");
    Py_DECREF(io);
    if (!open_fn) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *args = PyTuple_Pack(2, path, PyUnicode_FromString("w"));
    PyObject *file = PyObject_CallObject(open_fn, args);
    Py_DECREF(args);
    Py_DECREF(open_fn);
    if (!file) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *write_fn = PyObject_GetAttrString(file, "write");
    if (!write_fn) { PyErr_Clear(); Py_DECREF(file); Py_RETURN_NONE; }
    PyObject *text = content ? PyObject_Str(content) : PyUnicode_FromString("null");
    PyObject *write_args = PyTuple_Pack(1, text ? text : PyUnicode_FromString(""));
    PyObject *write_res = PyObject_CallObject(write_fn, write_args);
    Py_XDECREF(write_res);
    Py_DECREF(write_args);
    Py_XDECREF(text);
    Py_DECREF(write_fn);
    PyObject *close_fn = PyObject_GetAttrString(file, "close");
    if (close_fn) {
        PyObject_CallObject(close_fn, NULL);
        Py_DECREF(close_fn);
    }
    Py_DECREF(file);
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_import(PyObject *name) {
    if (!name) { Py_RETURN_NONE; }
    PyObject *module = PyImport_Import(name);
    if (!module) { PyErr_Clear(); Py_RETURN_NONE; }
    return module;
}

PyObject *zexus_cabi_int_from_long(long long value) {
    return PyLong_FromLongLong(value);
}

static PyObject *zexus_int_from_long(PyObject *self, PyObject *args) {
    long long value = 0;
    if (!PyArg_ParseTuple(args, "L", &value)) {
        return NULL;
    }
    return zexus_cabi_int_from_long(value);
}

PyObject *zexus_cabi_hash_block(PyObject *obj) {
    PyObject *bytes = zx_to_bytes(obj);
    if (!bytes) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *hex = zx_sha256_hex(bytes);
    Py_DECREF(bytes);
    return hex ? hex : Py_None;
}

PyObject *zexus_cabi_merkle_root_from_array(PyObject **items, Py_ssize_t count) {
    if (count <= 0) {
        return PyUnicode_FromString("");
    }
    PyObject *hashes = PyList_New(0);
    if (!hashes) { Py_RETURN_NONE; }
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *leaf = items[i] ? items[i] : Py_None;
        PyObject *bytes = zx_to_bytes(leaf);
        if (!bytes) { PyErr_Clear(); continue; }
        PyObject *hex = zx_sha256_hex(bytes);
        Py_DECREF(bytes);
        if (!hex) { PyErr_Clear(); continue; }
        PyList_Append(hashes, hex);
        Py_DECREF(hex);
    }
    while (PyList_Size(hashes) > 1) {
        Py_ssize_t n = PyList_Size(hashes);
        if (n % 2 != 0) {
            PyObject *last = PyList_GetItem(hashes, n - 1);
            PyList_Append(hashes, last);
            n += 1;
        }
        PyObject *new_hashes = PyList_New(0);
        for (Py_ssize_t i = 0; i < n; i += 2) {
            PyObject *h1 = PyList_GetItem(hashes, i);
            PyObject *h2 = PyList_GetItem(hashes, i + 1);
            PyObject *combined = PyUnicode_Concat(h1, h2);
            if (!combined) { PyErr_Clear(); continue; }
            PyObject *bytes = PyUnicode_AsEncodedString(combined, "utf-8", "strict");
            Py_DECREF(combined);
            if (!bytes) { PyErr_Clear(); continue; }
            PyObject *hex = zx_sha256_hex(bytes);
            Py_DECREF(bytes);
            if (!hex) { PyErr_Clear(); continue; }
            PyList_Append(new_hashes, hex);
            Py_DECREF(hex);
        }
        Py_DECREF(hashes);
        hashes = new_hashes;
    }
    PyObject *result = PyList_Size(hashes) > 0 ? PyList_GetItem(hashes, 0) : PyUnicode_FromString("");
    Py_XINCREF(result);
    Py_DECREF(hashes);
    return result;
}

PyObject *zexus_cabi_verify_signature(PyObject *env, PyObject *builtins, PyObject *sig, PyObject *msg, PyObject *pk) {
    PyObject *verify = NULL;
    if (builtins && PyDict_Check(builtins)) {
        verify = PyDict_GetItemString(builtins, "verify_sig");
    }
    if (!verify && env && PyDict_Check(env)) {
        verify = PyDict_GetItemString(env, "verify_sig");
    }
    if (verify && PyCallable_Check(verify)) {
        PyObject *res = PyObject_CallFunctionObjArgs(verify, sig, msg, pk, NULL);
        if (!res) { PyErr_Clear(); Py_RETURN_FALSE; }
        return res;
    }
    PyObject *bytes = zx_to_bytes(msg);
    if (!bytes) { PyErr_Clear(); Py_RETURN_FALSE; }
    PyObject *expected = zx_sha256_hex(bytes);
    Py_DECREF(bytes);
    if (!expected) { PyErr_Clear(); Py_RETURN_FALSE; }
    int eq = PyObject_RichCompareBool(sig, expected, Py_EQ);
    Py_DECREF(expected);
    if (eq < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (eq) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}

PyObject *zexus_cabi_state_read(PyObject *env, PyObject *key) {
    PyObject *state = zx_env_get_dict(env, "_blockchain_state", 1);
    if (!state || !PyDict_Check(state) || !key) { Py_RETURN_NONE; }
    PyObject *val = PyDict_GetItem(state, key);
    if (!val) Py_RETURN_NONE;
    Py_INCREF(val);
    return val;
}

PyObject *zexus_cabi_state_write(PyObject *env, PyObject *key, PyObject *value) {
    if (!key) { Py_RETURN_NONE; }
    PyObject *in_tx = zx_env_get_bool(env, "_in_transaction");
    int in_transaction = in_tx ? PyObject_IsTrue(in_tx) : 0;
    PyObject *target = NULL;
    if (in_transaction) {
        target = zx_env_get_dict(env, "_tx_pending_state", 1);
    } else {
        target = zx_env_get_dict(env, "_blockchain_state", 1);
    }
    if (target && PyDict_Check(target)) {
        PyDict_SetItem(target, key, value ? value : Py_None);
    }
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_tx_begin(PyObject *env) {
    if (!env || !PyDict_Check(env)) { Py_RETURN_NONE; }
    zx_env_set(env, "_in_transaction", Py_True);
    PyObject *pending = PyDict_New();
    if (pending) {
        zx_env_set(env, "_tx_pending_state", pending);
        Py_DECREF(pending);
    }
    PyObject *state = zx_env_get_dict(env, "_blockchain_state", 1);
    if (state) {
        PyObject *snapshot = PyDict_Copy(state);
        if (snapshot) {
            zx_env_set(env, "_tx_snapshot", snapshot);
            Py_DECREF(snapshot);
        }
    }
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_tx_commit(PyObject *env) {
    if (!env || !PyDict_Check(env)) { Py_RETURN_NONE; }
    PyObject *in_tx = zx_env_get_bool(env, "_in_transaction");
    int in_transaction = in_tx ? PyObject_IsTrue(in_tx) : 0;
    if (!in_transaction) { Py_RETURN_NONE; }
    PyObject *state = zx_env_get_dict(env, "_blockchain_state", 1);
    PyObject *pending = zx_env_get_dict(env, "_tx_pending_state", 0);
    if (state && pending && PyDict_Check(state) && PyDict_Check(pending)) {
        PyDict_Update(state, pending);
    }
    zx_env_set(env, "_in_transaction", Py_False);
    PyObject *empty = PyDict_New();
    if (empty) {
        zx_env_set(env, "_tx_pending_state", empty);
        Py_DECREF(empty);
    }
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_tx_revert(PyObject *env) {
    if (!env || !PyDict_Check(env)) { Py_RETURN_NONE; }
    PyObject *in_tx = zx_env_get_bool(env, "_in_transaction");
    int in_transaction = in_tx ? PyObject_IsTrue(in_tx) : 0;
    if (!in_transaction) { Py_RETURN_NONE; }
    PyObject *snapshot = zx_env_get_dict(env, "_tx_snapshot", 0);
    if (snapshot && PyDict_Check(snapshot)) {
        zx_env_set(env, "_blockchain_state", snapshot);
    }
    zx_env_set(env, "_in_transaction", Py_False);
    PyObject *empty = PyDict_New();
    if (empty) {
        zx_env_set(env, "_tx_pending_state", empty);
        Py_DECREF(empty);
    }
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_gas_charge(PyObject *env, PyObject *amount) {
    if (!env || !PyDict_Check(env)) return NULL;
    PyObject *k = PyUnicode_FromString("_gas_remaining");
    if (!k) return NULL;
    PyObject *cur = PyDict_GetItem(env, k);
    Py_DECREF(k);
    if (!cur) return NULL;
    if (!PyFloat_Check(cur) && !PyLong_Check(cur)) return NULL;
    if (PyFloat_Check(cur) && PyFloat_AsDouble(cur) == Py_HUGE_VAL) return NULL;
    PyObject *zero = PyLong_FromLong(0);
    PyObject *subtrahend = amount ? amount : zero;
    PyObject *new_gas = PyNumber_Subtract(cur, subtrahend);
    Py_DECREF(zero);
    if (!new_gas) { PyErr_Clear(); return NULL; }
    PyObject *cmp_zero = PyLong_FromLong(0);
    int neg = PyObject_RichCompareBool(new_gas, cmp_zero, Py_LT);
    Py_DECREF(cmp_zero);
    if (neg) {
        Py_DECREF(new_gas);
        PyObject *in_tx = zx_env_get_bool(env, "_in_transaction");
        int in_transaction = in_tx ? PyObject_IsTrue(in_tx) : 0;
        if (in_transaction) {
            PyObject *snapshot = zx_env_get_dict(env, "_tx_snapshot", 0);
            if (snapshot && PyDict_Check(snapshot)) {
                zx_env_set(env, "_blockchain_state", snapshot);
            }
            zx_env_set(env, "_in_transaction", Py_False);
        }
        PyObject *err = PyDict_New();
        if (!err) return NULL;
        PyDict_SetItemString(err, "error", PyUnicode_FromString("OutOfGas"));
        if (amount) {
            PyDict_SetItemString(err, "required", amount);
        } else {
            PyObject *zero_req = PyLong_FromLong(0);
            if (zero_req) {
                PyDict_SetItemString(err, "required", zero_req);
                Py_DECREF(zero_req);
            }
        }
        PyDict_SetItemString(err, "remaining", cur);
        return err;
    }
    PyObject *gas_key = PyUnicode_FromString("_gas_remaining");
    if (gas_key) {
        PyDict_SetItem(env, gas_key, new_gas);
        Py_DECREF(gas_key);
    }
    Py_DECREF(new_gas);
    return NULL;
}

PyObject *zexus_cabi_require(PyObject *env, PyObject *condition, PyObject *message) {
    int ok = condition ? PyObject_IsTrue(condition) : 0;
    if (ok > 0) {
        return NULL;
    }
    if (ok < 0) {
        PyErr_Clear();
    }

    if (env && PyDict_Check(env)) {
        PyObject *in_tx = zx_env_get_bool(env, "_in_transaction");
        int in_transaction = in_tx ? PyObject_IsTrue(in_tx) : 0;
        if (in_transaction) {
            PyObject *snapshot = zx_env_get_dict(env, "_tx_snapshot", 0);
            if (snapshot && PyDict_Check(snapshot)) {
                zx_env_set(env, "_blockchain_state", snapshot);
            }
            zx_env_set(env, "_in_transaction", Py_False);
            PyObject *empty = PyDict_New();
            if (empty) {
                zx_env_set(env, "_tx_pending_state", empty);
                Py_DECREF(empty);
            }
        }
    }

    PyObject *err = PyDict_New();
    if (!err) return NULL;
    PyDict_SetItemString(err, "error", PyUnicode_FromString("RequirementFailed"));
    PyObject *msg_obj = message && message != Py_None ? PyObject_Str(message) : PyUnicode_FromString("Requirement failed");
    if (msg_obj) {
        PyDict_SetItemString(err, "message", msg_obj);
        Py_DECREF(msg_obj);
    }
    return err;
}

PyObject *zexus_cabi_ledger_append(PyObject *env, PyObject *entry) {
    PyObject *ledger = zx_env_get_list(env, "_ledger", 1);
    if (!ledger || !PyList_Check(ledger)) { Py_RETURN_NONE; }
    if (entry && PyDict_Check(entry)) {
        PyObject *ts_key = PyUnicode_FromString("timestamp");
        int has_ts = ts_key ? PyDict_Contains(entry, ts_key) : 0;
        Py_XDECREF(ts_key);
        if (!has_ts) {
            PyObject *time_mod = PyImport_ImportModule("time");
            if (time_mod) {
                PyObject *time_fn = PyObject_GetAttrString(time_mod, "time");
                Py_DECREF(time_mod);
                if (time_fn) {
                    PyObject *ts = PyObject_CallObject(time_fn, NULL);
                    Py_DECREF(time_fn);
                    if (ts) {
                        PyDict_SetItemString(entry, "timestamp", ts);
                        Py_DECREF(ts);
                    }
                }
            }
        }
    }
    if (ledger && PyList_Check(ledger)) {
        PyList_Append(ledger, entry ? entry : Py_None);
    }
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_register_event(PyObject *vm, PyObject *name, PyObject *handler) {
    if (!vm || !name) { Py_RETURN_NONE; }
    PyObject *events = zx_get_attr(vm, "_events");
    if (!events || !PyDict_Check(events)) { Py_XDECREF(events); Py_RETURN_NONE; }
    PyObject *lst = PyDict_GetItem(events, name);
    if (!lst) {
        lst = PyList_New(0);
        if (lst) {
            PyDict_SetItem(events, name, lst);
            Py_DECREF(lst);
        }
    }
    if (handler && handler != Py_None && lst && PyList_Check(lst)) {
        int contains = PySequence_Contains(lst, handler);
        if (contains == 0) {
            PyList_Append(lst, handler);
        }
    }
    Py_DECREF(events);
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_emit_event(PyObject *vm, PyObject *env, PyObject *builtins, PyObject *name, PyObject *payload) {
    if (!vm || !name) { Py_RETURN_NONE; }
    PyObject *events = zx_get_attr(vm, "_events");
    if (!events || !PyDict_Check(events)) { Py_XDECREF(events); Py_RETURN_NONE; }
    PyObject *handlers = PyDict_GetItem(events, name);
    if (!handlers || !PyList_Check(handlers)) { Py_DECREF(events); Py_RETURN_NONE; }
    PyObject *builtins_dict = builtins;
    PyObject *env_dict = env;
    Py_ssize_t n = PyList_Size(handlers);
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *h = PyList_GetItem(handlers, i);
        PyObject *fn = NULL;
        if (h && PyUnicode_Check(h)) {
            if (builtins_dict && PyDict_Check(builtins_dict)) fn = PyDict_GetItem(builtins_dict, h);
            if (!fn && env_dict && PyDict_Check(env_dict)) fn = PyDict_GetItem(env_dict, h);
        } else {
            fn = h;
        }
        if (!fn) continue;
        PyObject *args_list = PyList_New(1);
        if (!args_list) continue;
        PyObject *payload_obj = payload ? payload : Py_None;
        Py_INCREF(payload_obj);
        PyList_SetItem(args_list, 0, payload_obj);
        PyObject *res = PyObject_CallMethod(vm, "_call_builtin_async_obj", "OOO", fn, args_list, Py_False);
        Py_DECREF(args_list);
        if (!res) { PyErr_Clear(); continue; }
        if (zx_is_awaitable(res)) {
            PyObject *awaited = PyObject_CallMethod(vm, "_run_coroutine_sync", "O", res);
            Py_XDECREF(awaited);
        }
        Py_DECREF(res);
    }
    Py_DECREF(events);
    Py_RETURN_NONE;
}

static PyObject *zx_spawn_task(PyObject *vm, PyObject *coro) {
    if (!vm || !coro) return NULL;
    PyObject *asyncio = PyImport_ImportModule("asyncio");
    PyObject *task = NULL;
    if (asyncio) {
        PyObject *loop = PyObject_CallMethod(asyncio, "get_running_loop", NULL);
        if (!loop) {
            PyErr_Clear();
        } else {
            task = PyObject_CallMethod(loop, "create_task", "O", coro);
            Py_DECREF(loop);
        }
        Py_DECREF(asyncio);
    }
    if (!task) {
        PyObject *res = PyObject_CallMethod(vm, "_run_coroutine_sync", "O", coro);
        task = res ? res : Py_None;
        Py_XINCREF(task);
    }

    PyObject *tasks = zx_get_attr(vm, "_tasks");
    PyObject *counter = zx_get_attr(vm, "_task_counter");
    long next_id = 1;
    if (counter && PyLong_Check(counter)) {
        next_id = PyLong_AsLong(counter) + 1;
    }
    PyObject *new_counter = PyLong_FromLong(next_id);
    if (new_counter) {
        PyObject_SetAttrString(vm, "_task_counter", new_counter);
        Py_DECREF(new_counter);
    }
    if (tasks && PyDict_Check(tasks)) {
        PyObject *tid = PyUnicode_FromFormat("task_%ld", next_id);
        if (tid) {
            PyDict_SetItem(tasks, tid, task);
            Py_DECREF(task);
            Py_DECREF(tasks);
            return tid;
        }
    }
    Py_XDECREF(tasks);
    Py_XDECREF(task);
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_spawn_name(PyObject *vm, PyObject *env, PyObject *builtins, PyObject *name, PyObject *args_tuple) {
    if (!vm || !name) { Py_RETURN_NONE; }
    PyObject *callable = NULL;
    if (env && PyDict_Check(env)) callable = PyDict_GetItem(env, name);
    if (!callable && builtins && PyDict_Check(builtins)) callable = PyDict_GetItem(builtins, name);
    if (!callable) { Py_RETURN_NONE; }
    PyObject *empty = NULL;
    PyObject *args_source = args_tuple ? args_tuple : (empty = PyTuple_New(0));
    PyObject *args_list = PySequence_List(args_source);
    Py_XDECREF(empty);
    if (!args_list) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *coro = PyObject_CallMethod(vm, "_to_coro", "OO", callable, args_list);
    Py_DECREF(args_list);
    if (!coro) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *tid = zx_spawn_task(vm, coro);
    Py_DECREF(coro);
    return tid ? tid : Py_None;
}

PyObject *zexus_cabi_spawn_call(PyObject *vm, PyObject *callable, PyObject *args_tuple) {
    if (!vm || !callable) { Py_RETURN_NONE; }
    PyObject *empty = NULL;
    PyObject *args_source = args_tuple ? args_tuple : (empty = PyTuple_New(0));
    PyObject *args_list = PySequence_List(args_source);
    Py_XDECREF(empty);
    if (!args_list) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *coro = PyObject_CallMethod(vm, "_to_coro", "OO", callable, args_list);
    Py_DECREF(args_list);
    if (!coro) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *tid = zx_spawn_task(vm, coro);
    Py_DECREF(coro);
    return tid ? tid : Py_None;
}

PyObject *zexus_cabi_await(PyObject *vm, PyObject *task_or_coro) {
    if (!vm || !task_or_coro) { Py_RETURN_NONE; }
    PyObject *tasks = zx_get_attr(vm, "_tasks");
    if (tasks && PyDict_Check(tasks) && PyUnicode_Check(task_or_coro)) {
        PyObject *obj = PyDict_GetItem(tasks, task_or_coro);
        if (obj) {
            if (zx_is_awaitable(obj)) {
                PyObject *res = PyObject_CallMethod(vm, "_run_coroutine_sync", "O", obj);
                Py_DECREF(tasks);
                return res ? res : Py_None;
            }
            Py_INCREF(obj);
            Py_DECREF(tasks);
            return obj;
        }
    }
    Py_XDECREF(tasks);
    if (zx_is_awaitable(task_or_coro)) {
        PyObject *res = PyObject_CallMethod(vm, "_run_coroutine_sync", "O", task_or_coro);
        return res ? res : Py_None;
    }
    Py_INCREF(task_or_coro);
    return task_or_coro;
}

PyObject *zexus_cabi_define_enum(PyObject *env, PyObject *name, PyObject *spec) {
    if (!env || !PyDict_Check(env) || !name) { Py_RETURN_NONE; }
    PyObject *enums = zx_env_get_dict(env, "enums", 1);
    if (enums && PyDict_Check(enums)) {
        PyDict_SetItem(enums, name, spec ? spec : Py_None);
    }
    PyDict_SetItem(env, name, spec ? spec : Py_None);
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_define_protocol(PyObject *env, PyObject *name, PyObject *spec) {
    if (!env || !PyDict_Check(env) || !name) { Py_RETURN_NONE; }
    PyObject *protocols = zx_env_get_dict(env, "protocols", 1);
    if (protocols && PyDict_Check(protocols)) {
        PyDict_SetItem(protocols, name, spec ? spec : Py_None);
    }
    PyDict_SetItem(env, name, spec ? spec : Py_None);
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_assert_protocol(PyObject *env, PyObject *name, PyObject *spec) {
    if (!env || !PyDict_Check(env) || !name || !spec) {
        PyObject *empty = PyList_New(0);
        return PyTuple_Pack(2, Py_False, empty ? empty : PyList_New(0));
    }
    PyObject *obj = PyDict_GetItem(env, name);
    PyObject *methods = PyDict_GetItemString(spec, "methods");
    PyObject *missing = PyList_New(0);
    int ok = 1;
    if (methods && PySequence_Check(methods)) {
        Py_ssize_t n = PySequence_Size(methods);
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *m = PySequence_GetItem(methods, i);
            if (!m) { PyErr_Clear(); continue; }
            int has = obj ? PyObject_HasAttr(obj, m) : 0;
            if (has <= 0) {
                ok = 0;
                PyList_Append(missing, m);
            }
            Py_DECREF(m);
        }
    }
    PyObject *ok_obj = ok ? Py_True : Py_False;
    Py_INCREF(ok_obj);
    PyObject *result = PyTuple_Pack(2, ok_obj, missing ? missing : PyList_New(0));
    Py_DECREF(ok_obj);
    Py_XDECREF(missing);
    return result;
}

PyObject *zexus_cabi_define_capability(PyObject *env, PyObject *name, PyObject *definition) {
    if (!env || !PyDict_Check(env) || !name) { Py_RETURN_NONE; }
    PyObject *caps = zx_env_get_dict(env, "_capabilities", 1);
    PyObject *key = zx_unwrap_value(name);
    if (caps && PyDict_Check(caps)) {
        PyDict_SetItem(caps, key, definition ? definition : Py_None);
    }
    Py_DECREF(key);
    Py_RETURN_NONE;
}

static PyObject *zx_get_or_create_set(PyObject *mapping, PyObject *key) {
    if (!mapping || !PyDict_Check(mapping) || !key) return NULL;
    PyObject *val = PyDict_GetItem(mapping, key);
    if (val && PySet_Check(val)) {
        return val;
    }
    PyObject *set_obj = PySet_New(NULL);
    if (!set_obj) return NULL;
    PyDict_SetItem(mapping, key, set_obj);
    Py_DECREF(set_obj);
    return PyDict_GetItem(mapping, key);
}

static PyObject *zx_to_string_key(PyObject *obj) {
    PyObject *val = zx_unwrap_value(obj);
    PyObject *s = PyObject_Str(val);
    Py_DECREF(val);
    if (!s) { PyErr_Clear(); return PyUnicode_FromString(""); }
    return s;
}

PyObject *zexus_cabi_grant_capability(PyObject *env, PyObject *entity, PyObject **caps, Py_ssize_t count) {
    if (!env || !PyDict_Check(env) || !entity) { Py_RETURN_NONE; }
    PyObject *grants = zx_env_get_dict(env, "_grants", 1);
    PyObject *entity_key = zx_to_string_key(entity);
    PyObject *entity_grants = zx_get_or_create_set(grants, entity_key);
    Py_DECREF(entity_key);
    if (entity_grants && PySet_Check(entity_grants)) {
        for (Py_ssize_t i = 0; i < count; i++) {
            PyObject *cap_key = zx_to_string_key(caps[i] ? caps[i] : Py_None);
            PySet_Add(entity_grants, cap_key);
            Py_DECREF(cap_key);
        }
    }
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_revoke_capability(PyObject *env, PyObject *entity, PyObject **caps, Py_ssize_t count) {
    if (!env || !PyDict_Check(env) || !entity) { Py_RETURN_NONE; }
    PyObject *grants = zx_env_get_dict(env, "_grants", 0);
    PyObject *entity_key = zx_to_string_key(entity);
    PyObject *entity_grants = grants ? PyDict_GetItem(grants, entity_key) : NULL;
    Py_DECREF(entity_key);
    if (entity_grants && PySet_Check(entity_grants)) {
        for (Py_ssize_t i = 0; i < count; i++) {
            PyObject *cap_key = zx_to_string_key(caps[i] ? caps[i] : Py_None);
            PySet_Discard(entity_grants, cap_key);
            Py_DECREF(cap_key);
        }
    }
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_audit_log(PyObject *env, PyObject *ts, PyObject *action, PyObject *data) {
    if (!env || !PyDict_Check(env)) { Py_RETURN_NONE; }
    PyObject *log_list = zx_env_get_list(env, "_audit_log", 1);
    if (!log_list || !PyList_Check(log_list)) { Py_RETURN_NONE; }
    PyObject *ts_val = zx_unwrap_value(ts ? ts : Py_None);
    PyObject *action_val = zx_unwrap_value(action ? action : Py_None);
    PyObject *data_val = zx_unwrap_value(data ? data : Py_None);
    PyObject *entry = PyDict_New();
    if (entry) {
        PyDict_SetItemString(entry, "timestamp", ts_val);
        PyDict_SetItemString(entry, "action", action_val);
        PyDict_SetItemString(entry, "data", data_val);
        PyList_Append(log_list, entry);
        Py_DECREF(entry);
    }
    Py_DECREF(ts_val);
    Py_DECREF(action_val);
    Py_DECREF(data_val);
    Py_RETURN_NONE;
}

static PyObject *zx_get_or_create_dict(PyObject *env, const char *key) {
    if (!env || !PyDict_Check(env) || !key) return NULL;
    PyObject *dict = zx_env_get_dict(env, key, 1);
    if (dict && PyDict_Check(dict)) return dict;
    return NULL;
}

PyObject *zexus_cabi_define_screen(PyObject *env, PyObject *name, PyObject *props) {
    PyObject *screens = zx_get_or_create_dict(env, "screens");
    if (!screens) { Py_RETURN_NONE; }
    PyObject *key = zx_to_string_key(name ? name : Py_None);
    PyDict_SetItem(screens, key, props ? props : Py_None);
    Py_DECREF(key);
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_define_component(PyObject *env, PyObject *name, PyObject *props) {
    PyObject *components = zx_get_or_create_dict(env, "components");
    if (!components) { Py_RETURN_NONE; }
    PyObject *key = zx_to_string_key(name ? name : Py_None);
    PyDict_SetItem(components, key, props ? props : Py_None);
    Py_DECREF(key);
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_define_theme(PyObject *env, PyObject *name, PyObject *props) {
    PyObject *themes = zx_get_or_create_dict(env, "themes");
    if (!themes) { Py_RETURN_NONE; }
    PyObject *key = zx_to_string_key(name ? name : Py_None);
    PyDict_SetItem(themes, key, props ? props : Py_None);
    Py_DECREF(key);
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_define_contract(PyObject **items, Py_ssize_t count, PyObject *name) {
    PyObject *members = PyDict_New();
    if (!members) { Py_RETURN_NONE; }
    Py_ssize_t total = count * 2;
    for (Py_ssize_t i = 0; i < count; i++) {
        Py_ssize_t key_index = total - 1 - (i * 2);
        Py_ssize_t val_index = total - 2 - (i * 2);
        PyObject *key_obj = (key_index >= 0 && key_index < total) ? items[key_index] : Py_None;
        PyObject *val_obj = (val_index >= 0 && val_index < total) ? items[val_index] : Py_None;
        PyObject *key_val = zx_unwrap_value(key_obj ? key_obj : Py_None);
        PyObject *key_str = PyObject_Str(key_val);
        Py_DECREF(key_val);
        if (key_str) {
            PyDict_SetItem(members, key_str, val_obj ? val_obj : Py_None);
            Py_DECREF(key_str);
        } else {
            PyErr_Clear();
        }
    }
    return members;
}

PyObject *zexus_cabi_define_entity(PyObject **items, Py_ssize_t count, PyObject *name) {
    PyObject *members = zexus_cabi_define_contract(items, count, name);
    if (!members || !PyDict_Check(members)) {
        return members ? members : Py_None;
    }
    PyObject *name_val = zx_unwrap_value(name ? name : Py_None);
    PyObject *name_str = PyObject_Str(name_val);
    Py_DECREF(name_val);
    if (!name_str) { PyErr_Clear(); name_str = PyUnicode_FromString(""); }
    PyDict_SetItemString(members, "_type", PyUnicode_FromString("entity"));
    PyDict_SetItemString(members, "_name", name_str);
    Py_DECREF(name_str);
    return members;
}

PyObject *zexus_cabi_restrict_access(PyObject *env, PyObject *obj, PyObject *prop, PyObject *restriction) {
    if (!env || !PyDict_Check(env)) { Py_RETURN_NONE; }
    PyObject *restrictions = zx_env_get_dict(env, "_restrictions", 1);
    if (!restrictions || !PyDict_Check(restrictions)) { Py_RETURN_NONE; }
    PyObject *obj_str = zx_to_string_key(obj ? obj : Py_None);
    PyObject *prop_str = prop ? zx_to_string_key(prop) : NULL;
    PyObject *key = NULL;
    if (prop_str && PyUnicode_GetLength(prop_str) > 0) {
        PyObject *dot = PyUnicode_FromString(".");
        PyObject *tmp = PyUnicode_Concat(obj_str, dot);
        Py_DECREF(dot);
        if (tmp) {
            key = PyUnicode_Concat(tmp, prop_str);
            Py_DECREF(tmp);
        }
    } else {
        key = obj_str;
        Py_INCREF(key);
    }
    if (key) {
        PyDict_SetItem(restrictions, key, restriction ? restriction : Py_None);
        Py_DECREF(key);
    }
    Py_DECREF(obj_str);
    Py_XDECREF(prop_str);
    Py_RETURN_NONE;
}

PyObject *zexus_cabi_enable_error_mode(PyObject *env) {
    if (!env || !PyDict_Check(env)) { Py_RETURN_NONE; }
    zx_env_set(env, "_continue_on_error", Py_True);
    Py_RETURN_NONE;
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

// Build a set from C array of PyObject*
static PyObject *zexus_build_set_from_array(PyObject *self, PyObject *args) {
    PyObject *addr_obj = NULL;
    Py_ssize_t count = 0;
    if (!PyArg_ParseTuple(args, "On", &addr_obj, &count)) {
        return NULL;
    }
    PyObject **items = (PyObject **)PyLong_AsVoidPtr(addr_obj);
    if (!items) {
        Py_RETURN_NONE;
    }
    return zexus_cabi_build_set_from_array(items, count);
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

static PyObject *zexus_export(PyObject *self, PyObject *args) {
    PyObject *env = NULL;
    PyObject *name = NULL;
    PyObject *value = Py_None;
    if (!PyArg_ParseTuple(args, "OO|O", &env, &name, &value)) {
        return NULL;
    }
    return zexus_cabi_export(env, name, value);
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
static PyObject *zexus_number_pow(PyObject *self, PyObject *args) {
    PyObject *a=NULL,*b=NULL; if (!PyArg_ParseTuple(args, "OO", &a, &b)) return NULL;
    return zexus_cabi_number_pow(a, b);
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

static PyObject *zexus_slice(PyObject *self, PyObject *args) {
    PyObject *obj=NULL,*start=NULL,*end=Py_None;
    if (!PyArg_ParseTuple(args, "OO|O", &obj, &start, &end)) return NULL;
    return zexus_cabi_slice(obj, start, end);
}

static PyObject *zexus_get_attr(PyObject *self, PyObject *args) {
    PyObject *obj=NULL,*attr=NULL; if (!PyArg_ParseTuple(args, "OO", &obj, &attr)) return NULL;
    return zexus_cabi_get_attr(obj, attr);
}

static PyObject *zexus_get_length(PyObject *self, PyObject *args) {
    PyObject *obj=NULL; if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
    return zexus_cabi_get_length(obj);
}

static PyObject *zexus_print(PyObject *self, PyObject *args) {
    PyObject *obj=NULL; if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
    return zexus_cabi_print(obj);
}

static PyObject *zexus_read(PyObject *self, PyObject *args) {
    PyObject *path=NULL; if (!PyArg_ParseTuple(args, "O", &path)) return NULL;
    return zexus_cabi_read(path);
}

static PyObject *zexus_write(PyObject *self, PyObject *args) {
    PyObject *path=NULL,*content=NULL;
    if (!PyArg_ParseTuple(args, "OO", &path, &content)) return NULL;
    return zexus_cabi_write(path, content);
}

static PyObject *zexus_import(PyObject *self, PyObject *args) {
    PyObject *name=NULL; if (!PyArg_ParseTuple(args, "O", &name)) return NULL;
    return zexus_cabi_import(name);
}

static PyObject *zexus_hash_block(PyObject *self, PyObject *args) {
    PyObject *obj=NULL; if (!PyArg_ParseTuple(args, "O", &obj)) return NULL;
    return zexus_cabi_hash_block(obj);
}

static PyObject *zexus_merkle_root_from_array(PyObject *self, PyObject *args) {
    PyObject *addr_obj=NULL; Py_ssize_t count=0;
    if (!PyArg_ParseTuple(args, "On", &addr_obj, &count)) return NULL;
    PyObject **items = (PyObject **)PyLong_AsVoidPtr(addr_obj);
    if (!items) Py_RETURN_NONE;
    return zexus_cabi_merkle_root_from_array(items, count);
}

static PyObject *zexus_verify_signature(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*builtins=NULL,*sig=NULL,*msg=NULL,*pk=NULL;
    if (!PyArg_ParseTuple(args, "OOOOO", &env, &builtins, &sig, &msg, &pk)) return NULL;
    return zexus_cabi_verify_signature(env, builtins, sig, msg, pk);
}

static PyObject *zexus_state_read(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*key=NULL; if (!PyArg_ParseTuple(args, "OO", &env, &key)) return NULL;
    return zexus_cabi_state_read(env, key);
}

static PyObject *zexus_state_write(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*key=NULL,*value=NULL; if (!PyArg_ParseTuple(args, "OOO", &env, &key, &value)) return NULL;
    return zexus_cabi_state_write(env, key, value);
}

static PyObject *zexus_tx_begin(PyObject *self, PyObject *args) {
    PyObject *env=NULL; if (!PyArg_ParseTuple(args, "O", &env)) return NULL;
    return zexus_cabi_tx_begin(env);
}

static PyObject *zexus_tx_commit(PyObject *self, PyObject *args) {
    PyObject *env=NULL; if (!PyArg_ParseTuple(args, "O", &env)) return NULL;
    return zexus_cabi_tx_commit(env);
}

static PyObject *zexus_tx_revert(PyObject *self, PyObject *args) {
    PyObject *env=NULL; if (!PyArg_ParseTuple(args, "O", &env)) return NULL;
    return zexus_cabi_tx_revert(env);
}

static PyObject *zexus_gas_charge(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*amount=NULL; if (!PyArg_ParseTuple(args, "OO", &env, &amount)) return NULL;
    return zexus_cabi_gas_charge(env, amount);
}

static PyObject *zexus_require(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*condition=NULL,*message=Py_None;
    if (!PyArg_ParseTuple(args, "OO|O", &env, &condition, &message)) return NULL;
    return zexus_cabi_require(env, condition, message);
}

static PyObject *zexus_ledger_append(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*entry=NULL; if (!PyArg_ParseTuple(args, "OO", &env, &entry)) return NULL;
    return zexus_cabi_ledger_append(env, entry);
}

static PyObject *zexus_register_event(PyObject *self, PyObject *args) {
    PyObject *vm=NULL,*name=NULL,*handler=NULL; if (!PyArg_ParseTuple(args, "OOO", &vm, &name, &handler)) return NULL;
    return zexus_cabi_register_event(vm, name, handler);
}

static PyObject *zexus_emit_event(PyObject *self, PyObject *args) {
    PyObject *vm=NULL,*env=NULL,*builtins=NULL,*name=NULL,*payload=NULL;
    if (!PyArg_ParseTuple(args, "OOOOO", &vm, &env, &builtins, &name, &payload)) return NULL;
    return zexus_cabi_emit_event(vm, env, builtins, name, payload);
}

static PyObject *zexus_spawn_name(PyObject *self, PyObject *args) {
    PyObject *vm=NULL,*env=NULL,*builtins=NULL,*name=NULL,*args_tuple=NULL;
    if (!PyArg_ParseTuple(args, "OOOOO", &vm, &env, &builtins, &name, &args_tuple)) return NULL;
    return zexus_cabi_spawn_name(vm, env, builtins, name, args_tuple);
}

static PyObject *zexus_spawn_call(PyObject *self, PyObject *args) {
    PyObject *vm=NULL,*callable=NULL,*args_tuple=NULL;
    if (!PyArg_ParseTuple(args, "OOO", &vm, &callable, &args_tuple)) return NULL;
    return zexus_cabi_spawn_call(vm, callable, args_tuple);
}

static PyObject *zexus_await(PyObject *self, PyObject *args) {
    PyObject *vm=NULL,*obj=NULL; if (!PyArg_ParseTuple(args, "OO", &vm, &obj)) return NULL;
    return zexus_cabi_await(vm, obj);
}

static PyObject *zexus_define_enum(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*name=NULL,*spec=NULL; if (!PyArg_ParseTuple(args, "OOO", &env, &name, &spec)) return NULL;
    return zexus_cabi_define_enum(env, name, spec);
}

static PyObject *zexus_define_protocol(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*name=NULL,*spec=NULL; if (!PyArg_ParseTuple(args, "OOO", &env, &name, &spec)) return NULL;
    return zexus_cabi_define_protocol(env, name, spec);
}

static PyObject *zexus_assert_protocol(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*name=NULL,*spec=NULL; if (!PyArg_ParseTuple(args, "OOO", &env, &name, &spec)) return NULL;
    return zexus_cabi_assert_protocol(env, name, spec);
}

static PyObject *zexus_define_capability(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*name=NULL,*definition=NULL; if (!PyArg_ParseTuple(args, "OOO", &env, &name, &definition)) return NULL;
    return zexus_cabi_define_capability(env, name, definition);
}

static PyObject *zexus_define_screen(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*name=NULL,*props=NULL; if (!PyArg_ParseTuple(args, "OOO", &env, &name, &props)) return NULL;
    return zexus_cabi_define_screen(env, name, props);
}

static PyObject *zexus_define_component(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*name=NULL,*props=NULL; if (!PyArg_ParseTuple(args, "OOO", &env, &name, &props)) return NULL;
    return zexus_cabi_define_component(env, name, props);
}

static PyObject *zexus_define_theme(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*name=NULL,*props=NULL; if (!PyArg_ParseTuple(args, "OOO", &env, &name, &props)) return NULL;
    return zexus_cabi_define_theme(env, name, props);
}

static PyObject *zexus_grant_capability(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*entity=NULL,*addr_obj=NULL; Py_ssize_t count=0;
    if (!PyArg_ParseTuple(args, "OOOn", &env, &entity, &addr_obj, &count)) return NULL;
    PyObject **items = (PyObject **)PyLong_AsVoidPtr(addr_obj);
    if (!items) Py_RETURN_NONE;
    return zexus_cabi_grant_capability(env, entity, items, count);
}

static PyObject *zexus_revoke_capability(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*entity=NULL,*addr_obj=NULL; Py_ssize_t count=0;
    if (!PyArg_ParseTuple(args, "OOOn", &env, &entity, &addr_obj, &count)) return NULL;
    PyObject **items = (PyObject **)PyLong_AsVoidPtr(addr_obj);
    if (!items) Py_RETURN_NONE;
    return zexus_cabi_revoke_capability(env, entity, items, count);
}

static PyObject *zexus_audit_log(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*ts=NULL,*action=NULL,*data=NULL; if (!PyArg_ParseTuple(args, "OOOO", &env, &ts, &action, &data)) return NULL;
    return zexus_cabi_audit_log(env, ts, action, data);
}

static PyObject *zexus_define_contract(PyObject *self, PyObject *args) {
    PyObject *addr_obj=NULL,*name=NULL; Py_ssize_t count=0;
    if (!PyArg_ParseTuple(args, "OnO", &addr_obj, &count, &name)) return NULL;
    PyObject **items = (PyObject **)PyLong_AsVoidPtr(addr_obj);
    if (!items) Py_RETURN_NONE;
    return zexus_cabi_define_contract(items, count, name);
}

static PyObject *zexus_define_entity(PyObject *self, PyObject *args) {
    PyObject *addr_obj=NULL,*name=NULL; Py_ssize_t count=0;
    if (!PyArg_ParseTuple(args, "OnO", &addr_obj, &count, &name)) return NULL;
    PyObject **items = (PyObject **)PyLong_AsVoidPtr(addr_obj);
    if (!items) Py_RETURN_NONE;
    return zexus_cabi_define_entity(items, count, name);
}

static PyObject *zexus_restrict_access(PyObject *self, PyObject *args) {
    PyObject *env=NULL,*obj=NULL,*prop=NULL,*restriction=NULL;
    if (!PyArg_ParseTuple(args, "OOOO", &env, &obj, &prop, &restriction)) return NULL;
    return zexus_cabi_restrict_access(env, obj, prop, restriction);
}

static PyObject *zexus_enable_error_mode(PyObject *self, PyObject *args) {
    PyObject *env=NULL; if (!PyArg_ParseTuple(args, "O", &env)) return NULL;
    return zexus_cabi_enable_error_mode(env);
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
    PyDict_SetItemString(symbols, "zexus_build_set", PyLong_FromVoidPtr((void *)&zexus_cabi_build_set_from_array));
    PyDict_SetItemString(symbols, "zexus_build_list_from_array", PyLong_FromVoidPtr((void *)&zexus_cabi_build_list_from_array));
    PyDict_SetItemString(symbols, "zexus_build_map_from_array", PyLong_FromVoidPtr((void *)&zexus_cabi_build_map_from_array));
    PyDict_SetItemString(symbols, "zexus_build_set_from_array", PyLong_FromVoidPtr((void *)&zexus_cabi_build_set_from_array));
    PyDict_SetItemString(symbols, "zexus_env_get", PyLong_FromVoidPtr((void *)&zexus_cabi_env_get));
    PyDict_SetItemString(symbols, "zexus_env_set", PyLong_FromVoidPtr((void *)&zexus_cabi_env_set));
    PyDict_SetItemString(symbols, "zexus_export", PyLong_FromVoidPtr((void *)&zexus_cabi_export));
    PyDict_SetItemString(symbols, "zexus_number_add", PyLong_FromVoidPtr((void *)&zexus_cabi_number_add));
    PyDict_SetItemString(symbols, "zexus_number_sub", PyLong_FromVoidPtr((void *)&zexus_cabi_number_sub));
    PyDict_SetItemString(symbols, "zexus_number_mul", PyLong_FromVoidPtr((void *)&zexus_cabi_number_mul));
    PyDict_SetItemString(symbols, "zexus_number_div", PyLong_FromVoidPtr((void *)&zexus_cabi_number_div));
    PyDict_SetItemString(symbols, "zexus_number_mod", PyLong_FromVoidPtr((void *)&zexus_cabi_number_mod));
    PyDict_SetItemString(symbols, "zexus_number_pow", PyLong_FromVoidPtr((void *)&zexus_cabi_number_pow));
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
    PyDict_SetItemString(symbols, "zexus_slice", PyLong_FromVoidPtr((void *)&zexus_cabi_slice));
    PyDict_SetItemString(symbols, "zexus_get_attr", PyLong_FromVoidPtr((void *)&zexus_cabi_get_attr));
    PyDict_SetItemString(symbols, "zexus_get_length", PyLong_FromVoidPtr((void *)&zexus_cabi_get_length));
    PyDict_SetItemString(symbols, "zexus_print", PyLong_FromVoidPtr((void *)&zexus_cabi_print));
    PyDict_SetItemString(symbols, "zexus_read", PyLong_FromVoidPtr((void *)&zexus_cabi_read));
    PyDict_SetItemString(symbols, "zexus_write", PyLong_FromVoidPtr((void *)&zexus_cabi_write));
    PyDict_SetItemString(symbols, "zexus_import", PyLong_FromVoidPtr((void *)&zexus_cabi_import));
    PyDict_SetItemString(symbols, "zexus_int_from_long", PyLong_FromVoidPtr((void *)&zexus_cabi_int_from_long));
    PyDict_SetItemString(symbols, "zexus_build_tuple_from_array", PyLong_FromVoidPtr((void *)&zexus_cabi_build_tuple_from_array));
    PyDict_SetItemString(symbols, "zexus_hash_block", PyLong_FromVoidPtr((void *)&zexus_cabi_hash_block));
    PyDict_SetItemString(symbols, "zexus_merkle_root", PyLong_FromVoidPtr((void *)&zexus_cabi_merkle_root_from_array));
    PyDict_SetItemString(symbols, "zexus_verify_signature", PyLong_FromVoidPtr((void *)&zexus_cabi_verify_signature));
    PyDict_SetItemString(symbols, "zexus_state_read", PyLong_FromVoidPtr((void *)&zexus_cabi_state_read));
    PyDict_SetItemString(symbols, "zexus_state_write", PyLong_FromVoidPtr((void *)&zexus_cabi_state_write));
    PyDict_SetItemString(symbols, "zexus_tx_begin", PyLong_FromVoidPtr((void *)&zexus_cabi_tx_begin));
    PyDict_SetItemString(symbols, "zexus_tx_commit", PyLong_FromVoidPtr((void *)&zexus_cabi_tx_commit));
    PyDict_SetItemString(symbols, "zexus_tx_revert", PyLong_FromVoidPtr((void *)&zexus_cabi_tx_revert));
    PyDict_SetItemString(symbols, "zexus_gas_charge", PyLong_FromVoidPtr((void *)&zexus_cabi_gas_charge));
    PyDict_SetItemString(symbols, "zexus_require", PyLong_FromVoidPtr((void *)&zexus_cabi_require));
    PyDict_SetItemString(symbols, "zexus_ledger_append", PyLong_FromVoidPtr((void *)&zexus_cabi_ledger_append));
    PyDict_SetItemString(symbols, "zexus_register_event", PyLong_FromVoidPtr((void *)&zexus_cabi_register_event));
    PyDict_SetItemString(symbols, "zexus_emit_event", PyLong_FromVoidPtr((void *)&zexus_cabi_emit_event));
    PyDict_SetItemString(symbols, "zexus_spawn_name", PyLong_FromVoidPtr((void *)&zexus_cabi_spawn_name));
    PyDict_SetItemString(symbols, "zexus_spawn_call", PyLong_FromVoidPtr((void *)&zexus_cabi_spawn_call));
    PyDict_SetItemString(symbols, "zexus_await", PyLong_FromVoidPtr((void *)&zexus_cabi_await));
    PyDict_SetItemString(symbols, "zexus_define_enum", PyLong_FromVoidPtr((void *)&zexus_cabi_define_enum));
    PyDict_SetItemString(symbols, "zexus_define_protocol", PyLong_FromVoidPtr((void *)&zexus_cabi_define_protocol));
    PyDict_SetItemString(symbols, "zexus_assert_protocol", PyLong_FromVoidPtr((void *)&zexus_cabi_assert_protocol));
    PyDict_SetItemString(symbols, "zexus_define_capability", PyLong_FromVoidPtr((void *)&zexus_cabi_define_capability));
    PyDict_SetItemString(symbols, "zexus_define_screen", PyLong_FromVoidPtr((void *)&zexus_cabi_define_screen));
    PyDict_SetItemString(symbols, "zexus_define_component", PyLong_FromVoidPtr((void *)&zexus_cabi_define_component));
    PyDict_SetItemString(symbols, "zexus_define_theme", PyLong_FromVoidPtr((void *)&zexus_cabi_define_theme));
    PyDict_SetItemString(symbols, "zexus_grant_capability", PyLong_FromVoidPtr((void *)&zexus_cabi_grant_capability));
    PyDict_SetItemString(symbols, "zexus_revoke_capability", PyLong_FromVoidPtr((void *)&zexus_cabi_revoke_capability));
    PyDict_SetItemString(symbols, "zexus_audit_log", PyLong_FromVoidPtr((void *)&zexus_cabi_audit_log));
    PyDict_SetItemString(symbols, "zexus_define_contract", PyLong_FromVoidPtr((void *)&zexus_cabi_define_contract));
    PyDict_SetItemString(symbols, "zexus_define_entity", PyLong_FromVoidPtr((void *)&zexus_cabi_define_entity));
    PyDict_SetItemString(symbols, "zexus_restrict_access", PyLong_FromVoidPtr((void *)&zexus_cabi_restrict_access));
    PyDict_SetItemString(symbols, "zexus_enable_error_mode", PyLong_FromVoidPtr((void *)&zexus_cabi_enable_error_mode));
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
    {"build_set_from_array", zexus_build_set_from_array, METH_VARARGS, "Build set from C array"},
    {"env_get", zexus_env_get, METH_VARARGS, "Get item from env"},
    {"env_set", zexus_env_set, METH_VARARGS, "Set item in env"},
    {"export", zexus_export, METH_VARARGS, "Export value to env"},
    {"number_add", zexus_number_add, METH_VARARGS, "Add numbers"},
    {"number_sub", zexus_number_sub, METH_VARARGS, "Subtract numbers"},
    {"number_mul", zexus_number_mul, METH_VARARGS, "Multiply numbers"},
    {"number_div", zexus_number_div, METH_VARARGS, "Divide numbers"},
    {"number_mod", zexus_number_mod, METH_VARARGS, "Modulo"},
    {"number_pow", zexus_number_pow, METH_VARARGS, "Power"},
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
    {"slice", zexus_slice, METH_VARARGS, "Slice"},
    {"get_attr", zexus_get_attr, METH_VARARGS, "Get attr"},
    {"get_length", zexus_get_length, METH_VARARGS, "Get length"},
    {"print", zexus_print, METH_VARARGS, "Print"},
    {"read", zexus_read, METH_VARARGS, "Read file"},
    {"write", zexus_write, METH_VARARGS, "Write file"},
    {"import", zexus_import, METH_VARARGS, "Import module"},
    {"int_from_long", zexus_int_from_long, METH_VARARGS, "Create int from long"},
    {"hash_block", zexus_hash_block, METH_VARARGS, "Hash block"},
    {"merkle_root", zexus_merkle_root_from_array, METH_VARARGS, "Merkle root from array"},
    {"verify_signature", zexus_verify_signature, METH_VARARGS, "Verify signature"},
    {"state_read", zexus_state_read, METH_VARARGS, "State read"},
    {"state_write", zexus_state_write, METH_VARARGS, "State write"},
    {"tx_begin", zexus_tx_begin, METH_VARARGS, "Transaction begin"},
    {"tx_commit", zexus_tx_commit, METH_VARARGS, "Transaction commit"},
    {"tx_revert", zexus_tx_revert, METH_VARARGS, "Transaction revert"},
    {"gas_charge", zexus_gas_charge, METH_VARARGS, "Gas charge"},
    {"require", zexus_require, METH_VARARGS, "Require condition"},
    {"ledger_append", zexus_ledger_append, METH_VARARGS, "Ledger append"},
    {"register_event", zexus_register_event, METH_VARARGS, "Register event"},
    {"emit_event", zexus_emit_event, METH_VARARGS, "Emit event"},
    {"spawn_name", zexus_spawn_name, METH_VARARGS, "Spawn named call"},
    {"spawn_call", zexus_spawn_call, METH_VARARGS, "Spawn callable"},
    {"await_task", zexus_await, METH_VARARGS, "Await task or coroutine"},
    {"define_enum", zexus_define_enum, METH_VARARGS, "Define enum"},
    {"define_protocol", zexus_define_protocol, METH_VARARGS, "Define protocol"},
    {"assert_protocol", zexus_assert_protocol, METH_VARARGS, "Assert protocol"},
    {"define_capability", zexus_define_capability, METH_VARARGS, "Define capability"},
    {"define_screen", zexus_define_screen, METH_VARARGS, "Define screen"},
    {"define_component", zexus_define_component, METH_VARARGS, "Define component"},
    {"define_theme", zexus_define_theme, METH_VARARGS, "Define theme"},
    {"grant_capability", zexus_grant_capability, METH_VARARGS, "Grant capability"},
    {"revoke_capability", zexus_revoke_capability, METH_VARARGS, "Revoke capability"},
    {"audit_log", zexus_audit_log, METH_VARARGS, "Audit log"},
    {"define_contract", zexus_define_contract, METH_VARARGS, "Define contract"},
    {"define_entity", zexus_define_entity, METH_VARARGS, "Define entity"},
    {"restrict_access", zexus_restrict_access, METH_VARARGS, "Restrict access"},
    {"enable_error_mode", zexus_enable_error_mode, METH_VARARGS, "Enable error mode"},
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
