# GUI Native Keywords - Phase 2

**Status**: Future Enhancement
**Phase**: 2 - Integrate INTO Zexus
**Priority**: Medium
**Current State**: SCREEN/COMPONENT primitives exist

## Overview

Enhance existing SCREEN/COMPONENT keywords for complete GUI applications.

## APP Keyword

```zexus
app MainWindow {
    window {
        title: "My App"
        size: {width: 800, height: 600}
        resizable: true
    }
    
    layout vertical {
        # Header
        text "Welcome" size: 24 weight: "bold"
        
        # Input
        input placeholder: "Enter text" bind: text_value
        
        # Button
        button "Submit" on_click: handle_submit
        
        # List
        list items: user_list {
            item {
                text display: item.name
            }
        }
    }
    
    action handle_submit() {
        print("Submitted: " + text_value)
    }
}
```

## Related Documentation

- [RENDERER_UI](./RENDERER_UI.md) - Current UI system
- [@zexus/gui Package (Phase 3)](../packages/ZEXUS_GUI_PACKAGE.md)
- [Ecosystem Strategy](../ECOSYSTEM_STRATEGY.md)

---

**Status**: Planned Enhancement
**Last Updated**: 2025-12-29
