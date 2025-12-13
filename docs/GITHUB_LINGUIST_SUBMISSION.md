# Submitting Zexus to GitHub Linguist

To get native Zexus syntax highlighting on GitHub, we need to submit the language to the official [github/linguist](https://github.com/github/linguist) repository.

## Current Workaround

We've mapped `.zx` files to Python syntax highlighting in `.gitattributes`:
```
*.zx linguist-language=Python linguist-detectable=true
```

This provides basic syntax highlighting on GitHub using Python's grammar, which is similar enough to Zexus to be readable.

## Submitting to GitHub Linguist

### Prerequisites

1. **TextMate Grammar** ✅ - We have `syntaxes/zexus.tmLanguage.json`
2. **Language Samples** ✅ - We have examples in `tests/` and `examples/`
3. **Documentation** ✅ - README.md and docs/

### Steps to Submit

1. **Fork github/linguist**
   ```bash
   git clone https://github.com/github/linguist.git
   cd linguist
   ```

2. **Add Language Definition**
   
   Edit `lib/linguist/languages.yml`:
   ```yaml
   Zexus:
     type: programming
     color: "#FF6B35"
     extensions:
       - ".zx"
     tm_scope: source.zexus
     ace_mode: python
     language_id: 549872365
   ```

3. **Add Grammar**
   
   Copy `syntaxes/zexus.tmLanguage.json` to:
   ```
   vendor/grammars/zexus.tmLanguage.json
   ```

4. **Add Samples**
   
   Copy example files to:
   ```
   samples/Zexus/
   ```
   
   Include 2-3 representative `.zx` files showing:
   - Basic syntax (variables, functions)
   - Advanced features (contracts, entities)
   - Security features (protect, verify)

5. **Update Grammar Configuration**
   
   Edit `grammars.yml`:
   ```yaml
   vendor/grammars/zexus.tmLanguage.json:
     language_id: 549872365
     scope: source.zexus
   ```

6. **Test Locally**
   ```bash
   bundle install
   bundle exec rake test
   ```

7. **Submit PR**
   - Create a branch: `git checkout -b add-zexus-language`
   - Commit changes: `git commit -m "Add Zexus language"`
   - Push and create PR on GitHub
   - Reference this repo in the PR description

### PR Template

```markdown
## Adding Zexus Programming Language

### Language Information
- **Name**: Zexus
- **Type**: Programming Language
- **Extensions**: `.zx`
- **Repository**: https://github.com/Zaidux/zexus-interpreter
- **Description**: A modern, security-first programming language with built-in blockchain support

### Why This Language Should Be Added
- Active development and growing community
- Unique features: policy-as-code, blockchain contracts, persistent memory
- Clear, well-documented syntax with TextMate grammar
- Real-world usage in smart contract development

### Checklist
- [x] Added language definition to `lib/linguist/languages.yml`
- [x] Added TextMate grammar to `vendor/grammars/`
- [x] Added sample files to `samples/Zexus/`
- [x] Updated `grammars.yml`
- [x] Tests pass locally
```

## Timeline

GitHub Linguist PRs typically take:
- **Review**: 1-2 weeks
- **Merge**: 2-4 weeks after approval
- **Deployment**: Next GitHub release (monthly)

**Total**: 1-3 months for native GitHub support

## Alternative: Browser Extension

While waiting for GitHub support, users can install a browser extension like [Refined GitHub](https://github.com/refined-github/refined-github) which can add custom syntax highlighting.

## Current Status

- ✅ VS Code syntax highlighting working
- ✅ Grammar validated and complete
- 🟡 GitHub uses Python highlighting (temporary)
- ⏳ GitHub Linguist submission (pending)

## Reference

- GitHub Linguist: https://github.com/github/linguist
- Contributing Guide: https://github.com/github/linguist/blob/master/CONTRIBUTING.md
- Language Guidelines: https://github.com/github/linguist/blob/master/CONTRIBUTING.md#adding-a-language
