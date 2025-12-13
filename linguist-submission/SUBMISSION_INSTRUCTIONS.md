# GitHub Linguist Submission for Zexus

This directory contains all files needed to submit Zexus to GitHub Linguist.

## Files Prepared

1. **languages.yml** - Language definition entry
2. **grammars.yml** - Grammar configuration entry
3. **zexus.tmLanguage.json** - TextMate grammar file
4. **samples/** - Sample code files

## Submission Steps

### 1. Fork and Clone Linguist

```bash
# Fork github/linguist on GitHub first, then:
git clone https://github.com/YOUR_USERNAME/linguist.git
cd linguist
git checkout -b add-zexus-language
```

### 2. Add Language Definition

Append the contents of `languages.yml` to `lib/linguist/languages.yml`:

```bash
cat /path/to/zexus-interpreter/linguist-submission/languages.yml >> lib/linguist/languages.yml
```

Or manually add this entry in alphabetical order in `lib/linguist/languages.yml`.

### 3. Add Grammar

```bash
cp /path/to/zexus-interpreter/linguist-submission/zexus.tmLanguage.json vendor/grammars/
```

### 4. Update Grammar Configuration

Append the contents of `grammars.yml` to `grammars.yml`:

```bash
cat /path/to/zexus-interpreter/linguist-submission/grammars.yml >> grammars.yml
```

### 5. Add Sample Files

```bash
mkdir -p samples/Zexus
cp /path/to/zexus-interpreter/linguist-submission/samples/*.zx samples/Zexus/
```

### 6. Run Tests

```bash
bundle install
bundle exec rake samples
bundle exec rake test
```

### 7. Commit and Push

```bash
git add .
git commit -m "Add Zexus language

Zexus is a modern, security-first programming language with built-in 
blockchain support, policy-as-code security, and persistent memory management.

Repository: https://github.com/Zaidux/zexus-interpreter
Extensions: .zx
Type: Programming language"

git push origin add-zexus-language
```

### 8. Create Pull Request

1. Go to https://github.com/YOUR_USERNAME/linguist
2. Click "New Pull Request"
3. Use this template:

---

**Pull Request Title:**
```
Add Zexus language
```

**Pull Request Body:**
```markdown
## Language Information

- **Name**: Zexus
- **Type**: Programming Language
- **File Extensions**: `.zx`
- **Repository**: https://github.com/Zaidux/zexus-interpreter
- **Website**: https://github.com/Zaidux/zexus-interpreter

## Description

Zexus is a modern, security-first programming language designed for:
- Smart contract development with built-in blockchain primitives
- Policy-as-code security with PROTECT/VERIFY/RESTRICT keywords
- Persistent memory management with automatic leak detection
- Dependency injection and module mocking for testing
- Reactive state management with WATCH

## Why Add This Language?

1. **Active Development**: Regular commits and growing feature set
2. **Unique Features**: Combines security, blockchain, and systems programming
3. **Real-world Usage**: Used for smart contract development and secure applications
4. **Complete Tooling**: VS Code extension, CLI tools, comprehensive documentation
5. **Growing Community**: Open source with active maintenance

## Language Characteristics

- **Syntax**: Similar to Python/JavaScript with blockchain extensions
- **Paradigm**: Multi-paradigm (imperative, functional, object-oriented)
- **Typing**: Static typing with type inference
- **Compilation**: Hybrid interpreter/compiler

## Sample Code

Basic syntax:
```zexus
let count = 42
const PI = 3.14159

action greet(name: string) -> string {
    return "Hello, " + name
}

for each i in range(1, 10) {
    print(greet("User " + string(i)))
}
```

Smart contract:
```zexus
contract Token {
    persistent storage balances: Map<Address, integer>
    
    action transfer(to: Address, amount: integer) -> boolean {
        require(balances[msg.sender] >= amount, "Insufficient balance")
        balances[msg.sender] = balances[msg.sender] - amount
        balances[to] = balances.get(to, 0) + amount
        emit Transfer(msg.sender, to, amount)
        return true
    }
}
```

## Checklist

- [x] Added language definition to `lib/linguist/languages.yml`
- [x] Added TextMate grammar to `vendor/grammars/`
- [x] Added sample files to `samples/Zexus/`
- [x] Updated `grammars.yml`
- [x] Tests pass locally
- [x] Language has unique file extension (`.zx`)
- [x] Grammar is properly formatted JSON
- [x] Samples demonstrate key language features

## Additional Information

- **Documentation**: Complete language guide at repository
- **Package Manager**: ZPM (Zexus Package Manager) in development
- **IDE Support**: VS Code extension available
- **License**: MIT
```

---

## Timeline

- **PR Submission**: Immediate
- **Review**: 1-2 weeks
- **Merge**: 2-4 weeks after approval
- **GitHub Deployment**: Next release (monthly)
- **Total**: 1-3 months for live on GitHub

## Reference Links

- Linguist Contributing: https://github.com/github/linguist/blob/master/CONTRIBUTING.md
- Language Guidelines: https://github.com/github/linguist/blob/master/CONTRIBUTING.md#adding-a-language
- Testing: https://github.com/github/linguist#testing

## Status

Ready for submission! All files are prepared and validated.
