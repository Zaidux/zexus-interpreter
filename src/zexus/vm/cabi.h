#pragma once
#include <Python.h>

// Stable data layout: all Zexus VM values are represented as PyObject*.
// Native JIT works purely with PyObject* and C ABI helpers.

typedef PyObject* ZxValue;

#ifdef __cplusplus
extern "C" {
#endif

ZxValue zexus_cabi_call_callable(ZxValue callable, ZxValue args_tuple);
ZxValue zexus_cabi_call_method(ZxValue obj, ZxValue method_name, ZxValue args_tuple);
ZxValue zexus_cabi_call_name(ZxValue env, ZxValue builtins, ZxValue name, ZxValue args_tuple);

ZxValue zexus_cabi_env_get(ZxValue env, ZxValue name);
int zexus_cabi_env_set(ZxValue env, ZxValue name, ZxValue value);

ZxValue zexus_cabi_build_list_from_array(ZxValue *items, Py_ssize_t count);
ZxValue zexus_cabi_build_map_from_array(ZxValue *items, Py_ssize_t count);
ZxValue zexus_cabi_build_tuple_from_array(ZxValue *items, Py_ssize_t count);

ZxValue zexus_cabi_number_add(ZxValue a, ZxValue b);
ZxValue zexus_cabi_number_sub(ZxValue a, ZxValue b);
ZxValue zexus_cabi_number_mul(ZxValue a, ZxValue b);
ZxValue zexus_cabi_number_div(ZxValue a, ZxValue b);
ZxValue zexus_cabi_number_mod(ZxValue a, ZxValue b);
ZxValue zexus_cabi_number_pow(ZxValue a, ZxValue b);
ZxValue zexus_cabi_number_neg(ZxValue a);

ZxValue zexus_cabi_compare_eq(ZxValue a, ZxValue b);
ZxValue zexus_cabi_compare_ne(ZxValue a, ZxValue b);
ZxValue zexus_cabi_compare_lt(ZxValue a, ZxValue b);
ZxValue zexus_cabi_compare_gt(ZxValue a, ZxValue b);
ZxValue zexus_cabi_compare_lte(ZxValue a, ZxValue b);
ZxValue zexus_cabi_compare_gte(ZxValue a, ZxValue b);

int zexus_cabi_truthy_int(ZxValue a);
ZxValue zexus_cabi_not(ZxValue a);
ZxValue zexus_cabi_bool_and(ZxValue a, ZxValue b);
ZxValue zexus_cabi_bool_or(ZxValue a, ZxValue b);

ZxValue zexus_cabi_index(ZxValue obj, ZxValue idx);
ZxValue zexus_cabi_get_attr(ZxValue obj, ZxValue attr);
ZxValue zexus_cabi_get_length(ZxValue obj);

ZxValue zexus_cabi_print(ZxValue obj);
ZxValue zexus_cabi_read(ZxValue path);
ZxValue zexus_cabi_import(ZxValue name);

ZxValue zexus_cabi_int_from_long(long long value);

ZxValue zexus_cabi_hash_block(ZxValue obj);
ZxValue zexus_cabi_merkle_root_from_array(ZxValue *items, Py_ssize_t count);
ZxValue zexus_cabi_verify_signature(ZxValue env, ZxValue builtins, ZxValue sig, ZxValue msg, ZxValue pk);

ZxValue zexus_cabi_state_read(ZxValue env, ZxValue key);
ZxValue zexus_cabi_state_write(ZxValue env, ZxValue key, ZxValue value);
ZxValue zexus_cabi_tx_begin(ZxValue env);
ZxValue zexus_cabi_tx_commit(ZxValue env);
ZxValue zexus_cabi_tx_revert(ZxValue env);
ZxValue zexus_cabi_gas_charge(ZxValue env, ZxValue amount);
ZxValue zexus_cabi_ledger_append(ZxValue env, ZxValue entry);

ZxValue zexus_cabi_register_event(ZxValue vm, ZxValue name, ZxValue handler);
ZxValue zexus_cabi_emit_event(ZxValue vm, ZxValue env, ZxValue builtins, ZxValue name, ZxValue payload);
ZxValue zexus_cabi_spawn_name(ZxValue vm, ZxValue env, ZxValue builtins, ZxValue name, ZxValue args_tuple);
ZxValue zexus_cabi_spawn_call(ZxValue vm, ZxValue callable, ZxValue args_tuple);
ZxValue zexus_cabi_await(ZxValue vm, ZxValue task_or_coro);

ZxValue zexus_cabi_define_enum(ZxValue env, ZxValue name, ZxValue spec);
ZxValue zexus_cabi_define_protocol(ZxValue env, ZxValue name, ZxValue spec);

#ifdef __cplusplus
}
#endif

typedef ZxValue (*zx_call_callable_fn)(ZxValue callable, ZxValue args_tuple);
typedef ZxValue (*zx_call_method_fn)(ZxValue obj, ZxValue method_name, ZxValue args_tuple);
typedef ZxValue (*zx_call_name_fn)(ZxValue env, ZxValue builtins, ZxValue name, ZxValue args_tuple);

typedef ZxValue (*zx_env_get_fn)(ZxValue env, ZxValue name);
typedef int (*zx_env_set_fn)(ZxValue env, ZxValue name, ZxValue value);

typedef ZxValue (*zx_build_list_from_array_fn)(ZxValue *items, Py_ssize_t count);
typedef ZxValue (*zx_build_map_from_array_fn)(ZxValue *items, Py_ssize_t count);
typedef ZxValue (*zx_build_tuple_from_array_fn)(ZxValue *items, Py_ssize_t count);

typedef ZxValue (*zx_number_binop_fn)(ZxValue a, ZxValue b);
typedef ZxValue (*zx_number_unop_fn)(ZxValue a);

typedef ZxValue (*zx_compare_fn)(ZxValue a, ZxValue b);

typedef int (*zx_truthy_int_fn)(ZxValue a);

typedef ZxValue (*zx_index_fn)(ZxValue obj, ZxValue idx);
typedef ZxValue (*zx_get_attr_fn)(ZxValue obj, ZxValue attr);
typedef ZxValue (*zx_get_length_fn)(ZxValue obj);
