#define PY_SSIZE_T_CLEAN
#include <Python.h>

typedef PyObject* ZxValue;

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
    if (!obj) return PyBytes_FromString("");
    if (PyBytes_Check(obj)) { Py_INCREF(obj); return obj; }
    if (PyUnicode_Check(obj)) return PyUnicode_AsEncodedString(obj, "utf-8", "strict");
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

ZxValue zexus_rt_export(ZxValue env, ZxValue name, ZxValue value) {
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

ZxValue zexus_rt_iter_next(ZxValue iterator) {
    if (!iterator) { return NULL; }
    PyObject *val = PyIter_Next(iterator);
    if (!val) {
        if (PyErr_Occurred()) {
            PyErr_Clear();
        }
        return NULL;
    }
    return val;
}

ZxValue zexus_rt_build_set(ZxValue *items, Py_ssize_t count) {
    PyObject *set = PySet_New(NULL);
    if (!set) return NULL;
    for (Py_ssize_t i = 0; i < count; i++) {
        PyObject *item = items[i] ? items[i] : Py_None;
        PySet_Add(set, item);
    }
    return set;
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
ZxValue zexus_rt_pow(ZxValue a, ZxValue b) { return PyNumber_Power(a, b, Py_None); }
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

ZxValue zexus_rt_slice(ZxValue obj, ZxValue start, ZxValue end) {
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

ZxValue zexus_rt_print(ZxValue obj) {
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

ZxValue zexus_rt_read(ZxValue path) {
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

ZxValue zexus_rt_write(ZxValue path, ZxValue content) {
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

ZxValue zexus_rt_import(ZxValue name) {
    if (!name) { Py_RETURN_NONE; }
    PyObject *module = PyImport_Import(name);
    if (!module) { PyErr_Clear(); Py_RETURN_NONE; }
    return module;
}

ZxValue zexus_rt_int_from_long(long long value) {
    return PyLong_FromLongLong(value);
}

ZxValue zexus_rt_hash_block(ZxValue obj) {
    PyObject *bytes = zx_to_bytes(obj);
    if (!bytes) { PyErr_Clear(); Py_RETURN_NONE; }
    PyObject *hex = zx_sha256_hex(bytes);
    Py_DECREF(bytes);
    return hex ? hex : Py_None;
}

ZxValue zexus_rt_merkle_root(ZxValue *items, Py_ssize_t count) {
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

ZxValue zexus_rt_verify_signature(ZxValue env, ZxValue builtins, ZxValue sig, ZxValue msg, ZxValue pk) {
    PyObject *verify = NULL;
    if (builtins && PyDict_Check(builtins)) verify = PyDict_GetItemString(builtins, "verify_sig");
    if (!verify && env && PyDict_Check(env)) verify = PyDict_GetItemString(env, "verify_sig");
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

ZxValue zexus_rt_state_read(ZxValue env, ZxValue key) {
    PyObject *state = zx_env_get_dict(env, "_blockchain_state", 1);
    if (!state || !PyDict_Check(state) || !key) { Py_RETURN_NONE; }
    PyObject *val = PyDict_GetItem(state, key);
    if (!val) Py_RETURN_NONE;
    Py_INCREF(val);
    return val;
}

ZxValue zexus_rt_state_write(ZxValue env, ZxValue key, ZxValue value) {
    if (!key) { Py_RETURN_NONE; }
    PyObject *in_tx = zx_env_get_bool(env, "_in_transaction");
    int in_transaction = in_tx ? PyObject_IsTrue(in_tx) : 0;
    PyObject *target = in_transaction ? zx_env_get_dict(env, "_tx_pending_state", 1)
                                       : zx_env_get_dict(env, "_blockchain_state", 1);
    if (target && PyDict_Check(target)) {
        PyDict_SetItem(target, key, value ? value : Py_None);
    }
    Py_RETURN_NONE;
}

ZxValue zexus_rt_tx_begin(ZxValue env) {
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

ZxValue zexus_rt_tx_commit(ZxValue env) {
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

ZxValue zexus_rt_tx_revert(ZxValue env) {
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

ZxValue zexus_rt_gas_charge(ZxValue env, ZxValue amount) {
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

ZxValue zexus_rt_require(ZxValue env, ZxValue condition, ZxValue message) {
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

ZxValue zexus_rt_ledger_append(ZxValue env, ZxValue entry) {
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
    PyList_Append(ledger, entry ? entry : Py_None);
    Py_RETURN_NONE;
}

ZxValue zexus_rt_register_event(ZxValue vm, ZxValue name, ZxValue handler) {
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
        if (contains == 0) PyList_Append(lst, handler);
    }
    Py_DECREF(events);
    Py_RETURN_NONE;
}

ZxValue zexus_rt_emit_event(ZxValue vm, ZxValue env, ZxValue builtins, ZxValue name, ZxValue payload) {
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

ZxValue zexus_rt_spawn_name(ZxValue vm, ZxValue env, ZxValue builtins, ZxValue name, ZxValue args_tuple) {
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

ZxValue zexus_rt_spawn_call(ZxValue vm, ZxValue callable, ZxValue args_tuple) {
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

ZxValue zexus_rt_await(ZxValue vm, ZxValue task_or_coro) {
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

static PyObject *zx_get_or_create_lock(PyObject *env, PyObject *key) {
    if (!env || !PyDict_Check(env) || !key) return NULL;
    PyObject *locks = zx_env_get_dict(env, "_locks", 1);
    if (!locks || !PyDict_Check(locks)) return NULL;
    PyObject *lock = PyDict_GetItem(locks, key);
    if (lock) return lock;

    PyObject *threading = PyImport_ImportModule("threading");
    if (!threading) { PyErr_Clear(); return NULL; }
    PyObject *lock_cls = PyObject_GetAttrString(threading, "Lock");
    Py_DECREF(threading);
    if (!lock_cls) { PyErr_Clear(); return NULL; }
    PyObject *new_lock = PyObject_CallObject(lock_cls, NULL);
    Py_DECREF(lock_cls);
    if (!new_lock) { PyErr_Clear(); return NULL; }
    PyDict_SetItem(locks, key, new_lock);
    Py_DECREF(new_lock);
    return PyDict_GetItem(locks, key);
}

ZxValue zexus_rt_lock_acquire(ZxValue env, ZxValue key) {
    if (!env || !key) { Py_RETURN_NONE; }
    PyObject *lock = zx_get_or_create_lock(env, key);
    if (!lock) { Py_RETURN_NONE; }
    PyObject *res = PyObject_CallMethod(lock, "acquire", NULL);
    if (!res) { PyErr_Clear(); Py_RETURN_NONE; }
    Py_DECREF(res);
    Py_RETURN_NONE;
}

ZxValue zexus_rt_lock_release(ZxValue env, ZxValue key) {
    if (!env || !key) { Py_RETURN_NONE; }
    PyObject *lock = zx_get_or_create_lock(env, key);
    if (!lock) { Py_RETURN_NONE; }
    PyObject *res = PyObject_CallMethod(lock, "release", NULL);
    if (!res) { PyErr_Clear(); Py_RETURN_NONE; }
    Py_DECREF(res);
    Py_RETURN_NONE;
}

ZxValue zexus_rt_barrier_wait(ZxValue barrier, ZxValue timeout) {
    if (!barrier) { Py_RETURN_NONE; }
    PyObject *res = NULL;
    if (timeout && timeout != Py_None) {
        res = PyObject_CallMethod(barrier, "wait", "O", timeout);
    } else {
        res = PyObject_CallMethod(barrier, "wait", NULL);
    }
    if (!res) { PyErr_Clear(); Py_RETURN_NONE; }
    return res;
}

static PyObject *zx_get_or_create_atomic_lock(PyObject *env) {
    if (!env || !PyDict_Check(env)) return NULL;
    PyObject *lock = PyDict_GetItemString(env, "_atomic_lock");
    if (lock) return lock;
    PyObject *threading = PyImport_ImportModule("threading");
    if (!threading) { PyErr_Clear(); return NULL; }
    PyObject *lock_cls = PyObject_GetAttrString(threading, "Lock");
    Py_DECREF(threading);
    if (!lock_cls) { PyErr_Clear(); return NULL; }
    PyObject *new_lock = PyObject_CallObject(lock_cls, NULL);
    Py_DECREF(lock_cls);
    if (!new_lock) { PyErr_Clear(); return NULL; }
    PyDict_SetItemString(env, "_atomic_lock", new_lock);
    Py_DECREF(new_lock);
    return PyDict_GetItemString(env, "_atomic_lock");
}

ZxValue zexus_rt_atomic_add(ZxValue env, ZxValue key, ZxValue delta) {
    if (!env || !PyDict_Check(env) || !key) { Py_RETURN_NONE; }
    PyObject *lock = zx_get_or_create_atomic_lock(env);
    if (!lock) { Py_RETURN_NONE; }
    PyObject *acq = PyObject_CallMethod(lock, "acquire", NULL);
    Py_XDECREF(acq);
    PyObject *state = zx_env_get_dict(env, "_atomic_state", 1);
    PyObject *current = state ? PyDict_GetItem(state, key) : NULL;
    if (!current) current = PyLong_FromLong(0);
    else Py_INCREF(current);
    PyObject *delta_val = delta ? delta : PyLong_FromLong(0);
    if (!delta) Py_DECREF(delta_val);
    PyObject *new_val = PyNumber_Add(current, delta_val);
    if (new_val && state) {
        PyDict_SetItem(state, key, new_val);
    }
    Py_DECREF(current);
    PyObject *rel = PyObject_CallMethod(lock, "release", NULL);
    Py_XDECREF(rel);
    if (!new_val) { PyErr_Clear(); Py_RETURN_NONE; }
    return new_val;
}

ZxValue zexus_rt_atomic_cas(ZxValue env, ZxValue key, ZxValue expected, ZxValue new_value) {
    if (!env || !PyDict_Check(env) || !key) { Py_RETURN_FALSE; }
    PyObject *lock = zx_get_or_create_atomic_lock(env);
    if (!lock) { Py_RETURN_FALSE; }
    PyObject *acq = PyObject_CallMethod(lock, "acquire", NULL);
    Py_XDECREF(acq);
    PyObject *state = zx_env_get_dict(env, "_atomic_state", 1);
    PyObject *current = state ? PyDict_GetItem(state, key) : NULL;
    int eq = PyObject_RichCompareBool(current ? current : Py_None, expected ? expected : Py_None, Py_EQ);
    if (eq > 0 && state) {
        PyDict_SetItem(state, key, new_value ? new_value : Py_None);
    }
    PyObject *rel = PyObject_CallMethod(lock, "release", NULL);
    Py_XDECREF(rel);
    if (eq < 0) { PyErr_Clear(); Py_RETURN_FALSE; }
    if (eq) Py_RETURN_TRUE; else Py_RETURN_FALSE;
}

ZxValue zexus_rt_get_iter(ZxValue obj) {
    if (!obj) { Py_RETURN_NONE; }
    PyObject *iter = PyObject_GetIter(obj);
    if (!iter) { PyErr_Clear(); Py_RETURN_NONE; }
    return iter;
}

ZxValue zexus_rt_iter_next_pair(ZxValue iterator) {
    if (!iterator) { Py_RETURN_NONE; }
    PyObject *value = PyIter_Next(iterator);
    if (!value) {
        if (PyErr_Occurred()) { PyErr_Clear(); }
        PyObject *none = Py_None; Py_INCREF(none);
        PyObject *flag = Py_False; Py_INCREF(flag);
        PyObject *tpl = PyTuple_Pack(2, none, flag);
        Py_DECREF(none); Py_DECREF(flag);
        return tpl ? tpl : Py_None;
    }
    PyObject *flag = Py_True; Py_INCREF(flag);
    PyObject *tpl = PyTuple_Pack(2, value, flag);
    Py_DECREF(value);
    Py_DECREF(flag);
    return tpl ? tpl : Py_None;
}

ZxValue zexus_rt_define_enum(ZxValue env, ZxValue name, ZxValue spec) {
    if (!env || !PyDict_Check(env) || !name) { Py_RETURN_NONE; }
    PyObject *enums = zx_env_get_dict(env, "enums", 1);
    if (enums && PyDict_Check(enums)) {
        PyDict_SetItem(enums, name, spec ? spec : Py_None);
    }
    PyDict_SetItem(env, name, spec ? spec : Py_None);
    Py_RETURN_NONE;
}

ZxValue zexus_rt_define_protocol(ZxValue env, ZxValue name, ZxValue spec) {
    if (!env || !PyDict_Check(env) || !name) { Py_RETURN_NONE; }
    PyObject *protocols = zx_env_get_dict(env, "protocols", 1);
    if (protocols && PyDict_Check(protocols)) {
        PyDict_SetItem(protocols, name, spec ? spec : Py_None);
    }
    PyDict_SetItem(env, name, spec ? spec : Py_None);
    Py_RETURN_NONE;
}

ZxValue zexus_rt_assert_protocol(ZxValue env, ZxValue name, ZxValue spec) {
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

ZxValue zexus_rt_define_capability(ZxValue env, ZxValue name, ZxValue definition) {
    if (!env || !PyDict_Check(env) || !name) { Py_RETURN_NONE; }
    PyObject *caps = zx_env_get_dict(env, "_capabilities", 1);
    PyObject *key = zx_unwrap_value(name);
    if (caps && PyDict_Check(caps)) {
        PyDict_SetItem(caps, key, definition ? definition : Py_None);
    }
    Py_DECREF(key);
    Py_RETURN_NONE;
}

static PyObject *zx_to_string_key(PyObject *obj);

ZxValue zexus_rt_define_screen(ZxValue env, ZxValue name, ZxValue props) {
    if (!env || !PyDict_Check(env)) { Py_RETURN_NONE; }
    PyObject *screens = zx_env_get_dict(env, "screens", 1);
    if (!screens || !PyDict_Check(screens)) { Py_RETURN_NONE; }
    PyObject *key = zx_to_string_key(name ? name : Py_None);
    PyDict_SetItem(screens, key, props ? props : Py_None);
    Py_DECREF(key);
    Py_RETURN_NONE;
}

ZxValue zexus_rt_define_component(ZxValue env, ZxValue name, ZxValue props) {
    if (!env || !PyDict_Check(env)) { Py_RETURN_NONE; }
    PyObject *components = zx_env_get_dict(env, "components", 1);
    if (!components || !PyDict_Check(components)) { Py_RETURN_NONE; }
    PyObject *key = zx_to_string_key(name ? name : Py_None);
    PyDict_SetItem(components, key, props ? props : Py_None);
    Py_DECREF(key);
    Py_RETURN_NONE;
}

ZxValue zexus_rt_define_theme(ZxValue env, ZxValue name, ZxValue props) {
    if (!env || !PyDict_Check(env)) { Py_RETURN_NONE; }
    PyObject *themes = zx_env_get_dict(env, "themes", 1);
    if (!themes || !PyDict_Check(themes)) { Py_RETURN_NONE; }
    PyObject *key = zx_to_string_key(name ? name : Py_None);
    PyDict_SetItem(themes, key, props ? props : Py_None);
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

ZxValue zexus_rt_grant_capability(ZxValue env, ZxValue entity, ZxValue *caps, Py_ssize_t count) {
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

ZxValue zexus_rt_revoke_capability(ZxValue env, ZxValue entity, ZxValue *caps, Py_ssize_t count) {
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

ZxValue zexus_rt_audit_log(ZxValue env, ZxValue ts, ZxValue action, ZxValue data) {
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

ZxValue zexus_rt_define_contract(ZxValue *items, Py_ssize_t count, ZxValue name) {
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

ZxValue zexus_rt_define_entity(ZxValue *items, Py_ssize_t count, ZxValue name) {
    PyObject *members = zexus_rt_define_contract(items, count, name);
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

ZxValue zexus_rt_restrict_access(ZxValue env, ZxValue obj, ZxValue prop, ZxValue restriction) {
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

ZxValue zexus_rt_enable_error_mode(ZxValue env) {
    if (!env || !PyDict_Check(env)) { Py_RETURN_NONE; }
    zx_env_set(env, "_continue_on_error", Py_True);
    Py_RETURN_NONE;
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
    PyDict_SetItemString(symbols, "zexus_export", PyLong_FromVoidPtr((void *)&zexus_rt_export));
    PyDict_SetItemString(symbols, "zexus_build_list", PyLong_FromVoidPtr((void *)&zexus_rt_build_list));
    PyDict_SetItemString(symbols, "zexus_build_map", PyLong_FromVoidPtr((void *)&zexus_rt_build_map));
    PyDict_SetItemString(symbols, "zexus_build_set", PyLong_FromVoidPtr((void *)&zexus_rt_build_set));
    PyDict_SetItemString(symbols, "zexus_build_list_from_array", PyLong_FromVoidPtr((void *)&zexus_rt_build_list));
    PyDict_SetItemString(symbols, "zexus_build_map_from_array", PyLong_FromVoidPtr((void *)&zexus_rt_build_map));
    PyDict_SetItemString(symbols, "zexus_build_set_from_array", PyLong_FromVoidPtr((void *)&zexus_rt_build_set));
    PyDict_SetItemString(symbols, "zexus_iter_next", PyLong_FromVoidPtr((void *)&zexus_rt_iter_next));
    PyDict_SetItemString(symbols, "zexus_build_tuple_from_array", PyLong_FromVoidPtr((void *)&zexus_rt_build_tuple));
    PyDict_SetItemString(symbols, "zexus_number_add", PyLong_FromVoidPtr((void *)&zexus_rt_add));
    PyDict_SetItemString(symbols, "zexus_number_sub", PyLong_FromVoidPtr((void *)&zexus_rt_sub));
    PyDict_SetItemString(symbols, "zexus_number_mul", PyLong_FromVoidPtr((void *)&zexus_rt_mul));
    PyDict_SetItemString(symbols, "zexus_number_div", PyLong_FromVoidPtr((void *)&zexus_rt_div));
    PyDict_SetItemString(symbols, "zexus_number_mod", PyLong_FromVoidPtr((void *)&zexus_rt_mod));
    PyDict_SetItemString(symbols, "zexus_number_pow", PyLong_FromVoidPtr((void *)&zexus_rt_pow));
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
    PyDict_SetItemString(symbols, "zexus_slice", PyLong_FromVoidPtr((void *)&zexus_rt_slice));
    PyDict_SetItemString(symbols, "zexus_get_attr", PyLong_FromVoidPtr((void *)&zexus_rt_get_attr));
    PyDict_SetItemString(symbols, "zexus_get_length", PyLong_FromVoidPtr((void *)&zexus_rt_get_length));
    PyDict_SetItemString(symbols, "zexus_print", PyLong_FromVoidPtr((void *)&zexus_rt_print));
    PyDict_SetItemString(symbols, "zexus_read", PyLong_FromVoidPtr((void *)&zexus_rt_read));
    PyDict_SetItemString(symbols, "zexus_write", PyLong_FromVoidPtr((void *)&zexus_rt_write));
    PyDict_SetItemString(symbols, "zexus_import", PyLong_FromVoidPtr((void *)&zexus_rt_import));
    PyDict_SetItemString(symbols, "zexus_int_from_long", PyLong_FromVoidPtr((void *)&zexus_rt_int_from_long));
    PyDict_SetItemString(symbols, "zexus_hash_block", PyLong_FromVoidPtr((void *)&zexus_rt_hash_block));
    PyDict_SetItemString(symbols, "zexus_merkle_root", PyLong_FromVoidPtr((void *)&zexus_rt_merkle_root));
    PyDict_SetItemString(symbols, "zexus_verify_signature", PyLong_FromVoidPtr((void *)&zexus_rt_verify_signature));
    PyDict_SetItemString(symbols, "zexus_state_read", PyLong_FromVoidPtr((void *)&zexus_rt_state_read));
    PyDict_SetItemString(symbols, "zexus_state_write", PyLong_FromVoidPtr((void *)&zexus_rt_state_write));
    PyDict_SetItemString(symbols, "zexus_tx_begin", PyLong_FromVoidPtr((void *)&zexus_rt_tx_begin));
    PyDict_SetItemString(symbols, "zexus_tx_commit", PyLong_FromVoidPtr((void *)&zexus_rt_tx_commit));
    PyDict_SetItemString(symbols, "zexus_tx_revert", PyLong_FromVoidPtr((void *)&zexus_rt_tx_revert));
    PyDict_SetItemString(symbols, "zexus_gas_charge", PyLong_FromVoidPtr((void *)&zexus_rt_gas_charge));
    PyDict_SetItemString(symbols, "zexus_require", PyLong_FromVoidPtr((void *)&zexus_rt_require));
    PyDict_SetItemString(symbols, "zexus_ledger_append", PyLong_FromVoidPtr((void *)&zexus_rt_ledger_append));
    PyDict_SetItemString(symbols, "zexus_register_event", PyLong_FromVoidPtr((void *)&zexus_rt_register_event));
    PyDict_SetItemString(symbols, "zexus_emit_event", PyLong_FromVoidPtr((void *)&zexus_rt_emit_event));
    PyDict_SetItemString(symbols, "zexus_spawn_name", PyLong_FromVoidPtr((void *)&zexus_rt_spawn_name));
    PyDict_SetItemString(symbols, "zexus_spawn_call", PyLong_FromVoidPtr((void *)&zexus_rt_spawn_call));
    PyDict_SetItemString(symbols, "zexus_await", PyLong_FromVoidPtr((void *)&zexus_rt_await));
    PyDict_SetItemString(symbols, "zexus_lock_acquire", PyLong_FromVoidPtr((void *)&zexus_rt_lock_acquire));
    PyDict_SetItemString(symbols, "zexus_lock_release", PyLong_FromVoidPtr((void *)&zexus_rt_lock_release));
    PyDict_SetItemString(symbols, "zexus_barrier_wait", PyLong_FromVoidPtr((void *)&zexus_rt_barrier_wait));
    PyDict_SetItemString(symbols, "zexus_atomic_add", PyLong_FromVoidPtr((void *)&zexus_rt_atomic_add));
    PyDict_SetItemString(symbols, "zexus_atomic_cas", PyLong_FromVoidPtr((void *)&zexus_rt_atomic_cas));
    PyDict_SetItemString(symbols, "zexus_get_iter", PyLong_FromVoidPtr((void *)&zexus_rt_get_iter));
    PyDict_SetItemString(symbols, "zexus_iter_next_pair", PyLong_FromVoidPtr((void *)&zexus_rt_iter_next_pair));
    PyDict_SetItemString(symbols, "zexus_atomic_add", PyLong_FromVoidPtr((void *)&zexus_rt_atomic_add));
    PyDict_SetItemString(symbols, "zexus_atomic_cas", PyLong_FromVoidPtr((void *)&zexus_rt_atomic_cas));
    PyDict_SetItemString(symbols, "zexus_barrier_wait", PyLong_FromVoidPtr((void *)&zexus_rt_barrier_wait));
    PyDict_SetItemString(symbols, "zexus_define_enum", PyLong_FromVoidPtr((void *)&zexus_rt_define_enum));
    PyDict_SetItemString(symbols, "zexus_define_protocol", PyLong_FromVoidPtr((void *)&zexus_rt_define_protocol));
    PyDict_SetItemString(symbols, "zexus_assert_protocol", PyLong_FromVoidPtr((void *)&zexus_rt_assert_protocol));
    PyDict_SetItemString(symbols, "zexus_define_capability", PyLong_FromVoidPtr((void *)&zexus_rt_define_capability));
    PyDict_SetItemString(symbols, "zexus_define_screen", PyLong_FromVoidPtr((void *)&zexus_rt_define_screen));
    PyDict_SetItemString(symbols, "zexus_define_component", PyLong_FromVoidPtr((void *)&zexus_rt_define_component));
    PyDict_SetItemString(symbols, "zexus_define_theme", PyLong_FromVoidPtr((void *)&zexus_rt_define_theme));
    PyDict_SetItemString(symbols, "zexus_grant_capability", PyLong_FromVoidPtr((void *)&zexus_rt_grant_capability));
    PyDict_SetItemString(symbols, "zexus_revoke_capability", PyLong_FromVoidPtr((void *)&zexus_rt_revoke_capability));
    PyDict_SetItemString(symbols, "zexus_audit_log", PyLong_FromVoidPtr((void *)&zexus_rt_audit_log));
    PyDict_SetItemString(symbols, "zexus_define_contract", PyLong_FromVoidPtr((void *)&zexus_rt_define_contract));
    PyDict_SetItemString(symbols, "zexus_define_entity", PyLong_FromVoidPtr((void *)&zexus_rt_define_entity));
    PyDict_SetItemString(symbols, "zexus_restrict_access", PyLong_FromVoidPtr((void *)&zexus_rt_restrict_access));
    PyDict_SetItemString(symbols, "zexus_enable_error_mode", PyLong_FromVoidPtr((void *)&zexus_rt_enable_error_mode));
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
