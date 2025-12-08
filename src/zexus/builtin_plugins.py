"""
Builtin plugins for Zexus.

These are standard plugins shipped with the interpreter.
They provide commonly-used functionality as extensible modules.
"""

from src.zexus.plugin_system import PluginMetadata


# JSON Plugin
JSON_PLUGIN = PluginMetadata(
    name="json",
    version="1.0.0",
    author="Zexus Core Team",
    description="JSON serialization and deserialization",
    requires=[],
    provides=["json.parse", "json.stringify"],
    hooks=[],
    config={}
)

JSON_PLUGIN_CODE = """
@plugin {
  name: "json",
  version: "1.0.0",
  provides: ["json.parse", "json.stringify"],
  description: "JSON serialization support"
}

public action parse(input) {
  # Validate input is string
  if typeof(input) != "string" {
    ret error("JSON input must be string, got " ++ typeof(input));
  }
  
  # Call builtin JSON parser
  ret builtin_json_parse(input);
}

public action stringify(obj) {
  # Convert any value to JSON representation
  ret builtin_json_stringify(obj);
}

public action pretty(obj) {
  # Pretty-print JSON with indentation
  ret builtin_json_stringify(obj, { indent: 2 });
}
"""


# Logging Plugin
LOGGING_PLUGIN = PluginMetadata(
    name="logging",
    version="2.0.0",
    author="Zexus Core Team",
    description="Structured logging with configurable output",
    requires=[],
    provides=["logging.debug", "logging.info", "logging.warn", "logging.error"],
    hooks=[],
    config={
        "level": {"type": "string", "default": "info"},
        "format": {"type": "string", "default": "text"},
        "output": {"type": "string", "default": "stderr"}
    }
)

LOGGING_PLUGIN_CODE = """
@plugin {
  name: "logging",
  version: "2.0.0",
  provides: ["logging.debug", "logging.info", "logging.warn", "logging.error"],
  config: {
    level: { type: "string", default: "info" },
    format: { type: "string", default: "text" },
    output: { type: "string", default: "stderr" }
  }
}

let LEVELS = { debug: 0, info: 1, warn: 2, error: 3 };

action should_log(level) {
  let min_level = LEVELS[plugin.config.level] || 1;
  ret LEVELS[level] >= min_level;
}

action format_message(level, msg) {
  let ts = builtin_now();
  cond {
    plugin.config.format == "json" ? {
      ret builtin_json_stringify({
        level: level,
        message: msg,
        timestamp: ts
      });
    };
    else ? {
      ret "[" ++ level ++ "] " ++ msg ++ " (" ++ ts ++ ")";
    };
  }
}

public action debug(msg) {
  if should_log("debug") {
    let formatted = format_message("debug", msg);
    builtin_output(formatted, plugin.config.output);
  }
}

public action info(msg) {
  if should_log("info") {
    let formatted = format_message("info", msg);
    builtin_output(formatted, plugin.config.output);
  }
}

public action warn(msg) {
  if should_log("warn") {
    let formatted = format_message("warn", msg);
    builtin_output(formatted, plugin.config.output);
  }
}

public action error(msg) {
  if should_log("error") {
    let formatted = format_message("error", msg);
    builtin_output(formatted, plugin.config.output);
  }
}
"""


# Crypto Plugin
CRYPTO_PLUGIN = PluginMetadata(
    name="crypto",
    version="1.0.0",
    author="Zexus Core Team",
    description="Cryptographic functions (hash, HMAC, etc.)",
    requires=[],
    provides=["crypto.sha256", "crypto.sha512", "crypto.hmac", "crypto.random"],
    hooks=[],
    config={}
)

CRYPTO_PLUGIN_CODE = """
@plugin {
  name: "crypto",
  version: "1.0.0",
  provides: ["crypto.sha256", "crypto.sha512", "crypto.hmac", "crypto.random"]
}

public action sha256(input) {
  # Compute SHA256 hash of input
  ret builtin_crypto_hash("sha256", input);
}

public action sha512(input) {
  # Compute SHA512 hash of input
  ret builtin_crypto_hash("sha512", input);
}

public action hmac(key, msg, algorithm) {
  # Compute HMAC
  ret builtin_crypto_hmac(algorithm || "sha256", key, msg);
}

public action random(len) {
  # Generate cryptographically secure random bytes
  ret builtin_crypto_random(len || 32);
}
"""


# Validation Plugin
VALIDATION_PLUGIN = PluginMetadata(
    name="validation",
    version="1.0.0",
    author="Zexus Core Team",
    description="Data validation utilities",
    requires=[],
    provides=["validation.email", "validation.url", "validation.phone"],
    hooks=["type_validator"],
    config={}
)

VALIDATION_PLUGIN_CODE = """
@plugin {
  name: "validation",
  version: "1.0.0",
  provides: ["validation.email", "validation.url", "validation.phone"],
  hooks: ["type_validator"]
}

action register_type_validators() {
  plugin.register_hook("type_validator", action(value, spec) {
    cond {
      spec.type == "email" ? validate_email(value);
      spec.type == "url" ? validate_url(value);
      spec.type == "phone" ? validate_phone(value);
      else ? true;
    }
  });
}

public action validate_email(addr) {
  # Simple email validation regex
  ret typeof(addr) == "string" && 
      addr ~matches /^[^\\s@]+@[^\\s@]+\\.[^\\s@]+$/;
}

public action validate_url(url) {
  # Simple URL validation
  ret typeof(url) == "string" &&
      (url ~matches /^https?:\\/\\// || url ~matches /^www\\./);
}

public action validate_phone(phone) {
  # Simple phone validation (US format)
  ret typeof(phone) == "string" &&
      phone ~matches /^\\+?1?\\d{9,15}$/;
}

# Register validators on plugin load
register_type_validators();
"""


# Collections Plugin
COLLECTIONS_PLUGIN = PluginMetadata(
    name="collections",
    version="1.0.0",
    author="Zexus Core Team",
    description="Advanced collection utilities (map, filter, reduce, etc.)",
    requires=[],
    provides=["collections.map", "collections.filter", "collections.reduce", 
              "collections.zip", "collections.group"],
    hooks=[],
    config={}
)

COLLECTIONS_PLUGIN_CODE = """
@plugin {
  name: "collections",
  version: "1.0.0",
  provides: ["collections.map", "collections.filter", "collections.reduce",
             "collections.zip", "collections.group"]
}

public action map(arr, fn) {
  # Apply function to each array element
  let result = [];
  each item in arr {
    result = result ++ [fn(item)];
  }
  ret result;
}

public action filter(arr, predicate) {
  # Keep elements that satisfy predicate
  let result = [];
  each item in arr {
    if predicate(item) {
      result = result ++ [item];
    }
  }
  ret result;
}

public action reduce(arr, fn, initial) {
  # Fold array elements
  let acc = initial;
  each item in arr {
    acc = fn(acc, item);
  }
  ret acc;
}

public action zip(arr1, arr2) {
  # Combine two arrays
  let result = [];
  let len = min(len(arr1), len(arr2));
  let i = 0;
  while i < len {
    result = result ++ [[arr1[i], arr2[i]]];
    i = i + 1;
  }
  ret result;
}

public action group(arr, key_fn) {
  # Group elements by key function
  let groups = {};
  each item in arr {
    let key = key_fn(item);
    if !groups[key] {
      groups[key] = [];
    }
    groups[key] = groups[key] ++ [item];
  }
  ret groups;
}
"""


BUILTIN_PLUGINS = {
    "json": (JSON_PLUGIN, JSON_PLUGIN_CODE),
    "logging": (LOGGING_PLUGIN, LOGGING_PLUGIN_CODE),
    "crypto": (CRYPTO_PLUGIN, CRYPTO_PLUGIN_CODE),
    "validation": (VALIDATION_PLUGIN, VALIDATION_PLUGIN_CODE),
    "collections": (COLLECTIONS_PLUGIN, COLLECTIONS_PLUGIN_CODE),
}


def get_builtin_plugin(name: str):
    """Get a builtin plugin by name."""
    return BUILTIN_PLUGINS.get(name)
