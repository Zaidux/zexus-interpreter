# Zexus Standard Library

The Zexus standard library provides essential modules for common programming tasks.

## Modules

### fs - File System Operations
### http - HTTP Client
### json - JSON Parsing and Serialization  
### datetime - Date and Time Operations

## Installation

The standard library is included with Zexus. No additional installation required.

## Usage

Import modules in your Zexus code:

```zexus
use {fs} from "stdlib"
use {http} from "stdlib"
use {json} from "stdlib"
use {datetime} from "stdlib"
```

Or import specific functions:

```zexus
use {read_file, write_file} from "stdlib/fs"
use {get, post} from "stdlib/http"
use {parse, stringify} from "stdlib/json"
use {now, timestamp} from "stdlib/datetime"
```

## Module Documentation

See individual module documentation:
- [fs Module](FS_MODULE.md)
- [http Module](HTTP_MODULE.md)
- [json Module](JSON_MODULE.md)
- [datetime Module](DATETIME_MODULE.md)
