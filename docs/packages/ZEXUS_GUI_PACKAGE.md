# @zexus/gui Package Specification

**Status**: Planned (Phase 3)
**Dependencies**: SCREEN/COMPONENT Keywords, GUI Keywords (Phase 2)
**Priority**: Medium

## Overview

Cross-platform GUI framework built on Zexus UI primitives.

## Installation

```bash
zpm install @zexus/gui
```

## Features

- Cross-platform (Windows, macOS, Linux)
- Reactive UI updates
- Theming system
- Layout engine
- Event handling
- Animation support
- Native widgets

## Quick Start

```zexus
use {Application, Window, Button, Text} from "@zexus/gui"

let app = Application()
let window = Window({title: "My App", width: 800, height: 600})

let count = 0

let button = Button({
    text: "Increment",
    on_click: action() { count = count + 1 }
})

let label = Text({text: "Count: 0"})

watch count {
    label.text = "Count: " + string(count)
}

window.add(button)
window.add(label)

app.run(window)
```

---

**Status**: Planned
**Last Updated**: 2025-12-29
