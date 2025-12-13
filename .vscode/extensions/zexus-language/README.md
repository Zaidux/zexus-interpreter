# Zexus Language Extension for VS Code

This directory contains the VS Code language extension for Zexus.

## Installation

### Option 1: Workspace Settings (Current Setup)
The `.vscode/settings.json` file associates `.zx` files with the Zexus language and applies syntax highlighting.

### Option 2: Install Extension Globally

1. Package the extension:
```bash
cd .vscode/extensions/zexus-language
npm install -g vsce
vsce package
```

2. Install the .vsix file:
```bash
code --install-extension zexus-language-0.1.0.vsix
```

### Option 3: Development Mode

1. Open VS Code
2. Press `F5` to open Extension Development Host
3. The extension will be active in the new window

## Syntax Highlighting

Syntax highlighting is now enabled through:
- Workspace settings (`.vscode/settings.json`)
- TextMate grammar (`syntaxes/zexus.tmLanguage.json`)
- Language configuration (`language-configuration.json`)

## Features

- **Syntax Highlighting**: Keywords, strings, comments, operators
- **Auto-closing**: Brackets, quotes, braces
- **Code Snippets**: contract, entity, protect, watch, inject
- **Comment Toggling**: Use Ctrl+/ to toggle line comments
- **Folding**: Support for #region/#endregion

## Troubleshooting

If syntax highlighting doesn't work:

1. Reload VS Code window (Ctrl+Shift+P > "Reload Window")
2. Check file extension is `.zx`
3. Verify language mode in bottom-right corner shows "Zexus"
4. Check Output panel (View > Output > select "Extension Host")
