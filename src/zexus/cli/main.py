# src/zexus/cli/main.py
import click
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table

# Import your existing modules - UPDATED IMPORTS
from ..lexer import Lexer
from ..parser import Parser
# UPDATED: Import evaluate from evaluator package
from ..evaluator import evaluate
# UPDATED: Import Environment and String from object module
from ..object import Environment, String
from ..syntax_validator import SyntaxValidator
from ..hybrid_orchestrator import orchestrator
from ..config import config
# Import error handling
from ..error_reporter import get_error_reporter, ZexusError, print_error

console = Console()

def show_all_commands():
    """Display all available Zexus commands with descriptions"""
    console.print("\n[bold cyan]🚀 Zexus Programming Language - Available Commands[/bold cyan]\n")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Command", style="cyan", width=30)
    table.add_column("Description", style="white")
    
    commands = [
        ("zx run <file>", "Execute a Zexus program"),
        ("zx run --zexus", "Show this command list"),
        ("zx check <file>", "Check syntax with detailed validation"),
        ("zx validate <file>", "Validate and auto-fix syntax errors"),
        ("zx profile <file>", "Profile performance with time and memory tracking"),
        ("zx ast <file>", "Display Abstract Syntax Tree"),
        ("zx tokens <file>", "Show tokenization output"),
        ("zx repl", "Start interactive REPL"),
        ("zx init [name]", "Initialize new Zexus project"),
        ("zx debug <on|off|minimal|status>", "Control debug logging"),
        ("", ""),
        ("[bold]Global Options:[/bold]", ""),
        ("--syntax-style=<style>", "universal, tolerable, or auto (default)"),
        ("--execution-mode=<mode>", "interpreter, compiler, or auto (default)"),
        ("--advanced-parsing", "Enable multi-strategy parsing (default: on)"),
        ("--debug", "Enable debug output"),
        ("--version", "Show Zexus version"),
        ("--help", "Show detailed help"),
        ("", ""),
        ("[bold]Profile Options:[/bold]", ""),
        ("--memory/--no-memory", "Enable/disable memory profiling (default: on)"),
        ("--top N", "Show top N functions (default: 20)"),
        ("--json-output FILE", "Save profile data as JSON"),
        ("", ""),
        ("[bold]Examples:[/bold]", ""),
        ("zx run program.zx", "Run a program with auto-detection"),
        ("zx run --syntax-style=universal main.zx", "Run with strict syntax"),
        ("zx run --execution-mode=compiler fast.zx", "Force compiler mode"),
        ("zx check --debug program.zx", "Check syntax with debug info"),
        ("zx profile myapp.zx", "Profile with memory tracking"),
        ("zx profile --no-memory --top 10 app.zx", "Profile without memory, show top 10"),
        ("", ""),
        ("[bold]Built-in Functions:[/bold]", "100+ functions available"),
        ("", "Memory: persist_set, persist_get, track_memory"),
        ("", "Policy: protect, verify, restrict, sanitize"),
        ("", "DI: inject, register_dependency, mock_dependency"),
        ("", "Reactive: watch (keyword)"),
        ("", "Blockchain: transaction, emit, require, balance"),
        ("", "File System: read_file, write_file, mkdir, exists"),
        ("", "HTTP: http_get, http_post, http_put, http_delete"),
        ("", "JSON: json_parse, json_stringify, json_load, json_save"),
        ("", "DateTime: now, timestamp, format, add_days, diff_seconds"),
        ("", ""),
        ("[bold]Documentation:[/bold]", ""),
        ("", "README: github.com/Zaidux/zexus-interpreter"),
        ("", "Features: docs/features/ADVANCED_FEATURES_IMPLEMENTATION.md"),
        ("", "Dev Guide: src/README.md"),
        ("", "LSP Guide: docs/lsp/LSP_GUIDE.md"),
        ("", "Profiler: docs/profiler/PROFILER_GUIDE.md"),
        ("", "Stdlib: docs/stdlib/README.md"),
    ]
    
    for cmd, desc in commands:
        table.add_row(cmd, desc)
    
    console.print(table)
    console.print("\n[bold green]💡 Tip:[/bold green] Use 'zx <command> --help' for detailed command options\n")

@click.group(invoke_without_command=True)
@click.version_option(version="1.5.0", prog_name="Zexus")
@click.option('--syntax-style', type=click.Choice(['universal', 'tolerable', 'auto']),
              default='auto', help='Syntax style to use (universal=strict, tolerable=flexible)')
@click.option('--advanced-parsing', is_flag=True, default=True,
              help='Enable advanced multi-strategy parsing (recommended)')
@click.option('--execution-mode', type=click.Choice(['interpreter', 'compiler', 'auto']),
              default='auto', help='Execution engine to use')
@click.option('--debug', is_flag=True, help='Enable debug logging')
@click.option('--zexus', is_flag=True, help='Show all available Zexus commands')
@click.pass_context
def cli(ctx, syntax_style, advanced_parsing, execution_mode, debug, zexus):
    """Zexus Programming Language - Hybrid Interpreter/Compiler
    
    Use 'zx run --zexus' or 'zx --zexus' to see all available commands.
    """
    
    if zexus:
        show_all_commands()
        sys.exit(0)
    
    # If no command provided, show help
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        return
    
    ctx.ensure_object(dict)
    ctx.obj['SYNTAX_STYLE'] = syntax_style
    ctx.obj['ADVANCED_PARSING'] = advanced_parsing
    ctx.obj['EXECUTION_MODE'] = execution_mode
    ctx.obj['DEBUG'] = debug
    
    # Update config based on CLI flags
    if debug:
        config.enable_debug_logs = True
    if execution_mode == 'compiler':
        config.use_hybrid_compiler = True
    elif execution_mode == 'interpreter':
        config.use_hybrid_compiler = False

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.argument('args', nargs=-1)  # Accept any number of additional arguments
@click.pass_context
def run(ctx, file, args):
    """Run a Zexus program with hybrid execution"""
    # Register source for error reporting
    error_reporter = get_error_reporter()
    
    try:
        with open(file, 'r') as f:
            source_code = f.read()
        
        # Register source with error reporter
        error_reporter.register_source(file, source_code)
        
        syntax_style = ctx.obj['SYNTAX_STYLE']
        advanced_parsing = ctx.obj['ADVANCED_PARSING']
        execution_mode = ctx.obj['EXECUTION_MODE']
        validator = SyntaxValidator()
        
        console.print(f"🚀 [bold green]Running[/bold green] {file}")
        console.print(f"🔧 [bold blue]Execution mode:[/bold blue] {execution_mode}")
        console.print(f"📝 [bold blue]Syntax style:[/bold blue] {syntax_style}")
        console.print(f"🎯 [bold blue]Advanced parsing:[/bold blue] {'Enabled' if advanced_parsing else 'Disabled'}")
        
        # Auto-detect syntax style if needed
        if syntax_style == 'auto':
            syntax_style = validator.suggest_syntax_style(source_code)
            console.print(f"🔍 [bold blue]Detected syntax style:[/bold blue] {syntax_style}")
        
        # Validate syntax
        console.print("[dim]Validating syntax...[/dim]", end="")
        validation_result = validator.validate_code(source_code, syntax_style)
        console.print(" [green]done[/green]")
        if not validation_result['is_valid']:
            console.print(f"[bold yellow]⚠️  Syntax warnings: {validation_result['error_count']} issue(s) found[/bold yellow]")
            for suggestion in validation_result['suggestions']:
                severity_emoji = "❌" if suggestion['severity'] == 'error' else "⚠️"
                console.print(f"  {severity_emoji} Line {suggestion['line']}: {suggestion['message']}")

            # Auto-fix if there are errors
            if any(s['severity'] == 'error' for s in validation_result['suggestions']):
                console.print("[bold yellow]🛠️  Attempting auto-fix...[/bold yellow]")
                fixed_code, fix_result = validator.auto_fix(source_code, syntax_style)
                if fix_result['applied_fixes'] > 0:
                    console.print(f"✅ [bold green]Applied {fix_result['applied_fixes']} fixes[/bold green]")
                    source_code = fixed_code
                else:
                    console.print("[bold red]❌ Could not auto-fix errors, attempting to run anyway...[/bold red]")
        
        # Parse the program
        lexer = Lexer(source_code, filename=file)
        parser = Parser(lexer, syntax_style, enable_advanced_strategies=advanced_parsing)
        program = parser.parse_program()
        
        if parser.errors and any("critical" in e.lower() for e in parser.errors):
            console.print("[bold red]❌ Critical parser errors detected, cannot continue:[/bold red]")
            for error in parser.errors:
                console.print(f"  ❌ {error}")
            sys.exit(1)
        
        # Use the evaluator package
        env = Environment()
        
        # Set module context variables
        import os
        abs_file = os.path.abspath(file)
        env.set("__file__", String(abs_file))  # Absolute file path
        env.set("__FILE__", String(abs_file))  # Alternative name
        env.set("__MODULE__", String("__main__"))  # Main entry point
        env.set("__DIR__", String(os.path.dirname(abs_file)))  # Directory path
        
        # Set command-line arguments
        from ..object import List
        args_list = List([String(arg) for arg in args])
        env.set("__ARGS__", args_list)  # Command-line arguments
        env.set("__ARGV__", args_list)  # Alternative name
        
        # Detect package (if file is in a package structure)
        package_name = String("")
        try:
            rel_path = os.path.relpath(abs_file)
            if '/' in rel_path or '\\' in rel_path:
                parts = rel_path.replace('\\', '/').split('/')
                if len(parts) > 1:
                    package_name = String(parts[0])
        except (OSError, ValueError):
            pass  # Unable to determine package name
        env.set("__PACKAGE__", package_name)
        
        # UPDATED: Use the evaluate function from the evaluator package
        result = evaluate(program, env, debug_mode=ctx.obj['DEBUG'])
        
        if result and hasattr(result, 'inspect') and result.inspect() != 'null':
            console.print(f"\n✅ [bold green]Result:[/bold green] {result.inspect()}")
        elif isinstance(result, str) and result:
            console.print(f"\n📤 [bold blue]Output:[/bold blue] {result}")
        elif hasattr(result, 'value') and result.value is not None:
            console.print(f"\n📊 [bold blue]Result:[/bold blue] {result.value}")

    except ZexusError as e:
        # Handle our custom Zexus errors with nice formatting
        print_error(e)
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {str(e)}")
        if ctx.obj.get('DEBUG'):
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.pass_context
def check(ctx, file):
    """Check syntax of a Zexus file with detailed validation"""
    try:
        with open(file, 'r') as f:
            source_code = f.read()
        
        syntax_style = ctx.obj['SYNTAX_STYLE']
        advanced_parsing = ctx.obj['ADVANCED_PARSING']
        validator = SyntaxValidator()
        
        # Auto-detect syntax style if needed
        if syntax_style == 'auto':
            syntax_style = validator.suggest_syntax_style(source_code)
            console.print(f"🔍 [bold blue]Detected syntax style:[/bold blue] {syntax_style}")

        console.print(f"🔧 [bold blue]Advanced parsing:[/bold blue] {'Enabled' if advanced_parsing else 'Disabled'}")
        
        # Run syntax validation
        validation_result = validator.validate_code(source_code, syntax_style)
        
        # Also run parser for additional validation
        lexer = Lexer(source_code)
        parser = Parser(lexer, syntax_style, enable_advanced_strategies=advanced_parsing)
        program = parser.parse_program()
        
        # Display results
        if parser.errors or not validation_result['is_valid']:
            console.print("[bold red]❌ Syntax Issues Found:[/bold red]")
            
            # Show parser errors first
            for error in parser.errors:
                console.print(f"  🚫 Parser: {error}")
            
            # Show validator suggestions
            for suggestion in validation_result['suggestions']:
                severity_icon = "🚫" if suggestion['severity'] == 'error' else "⚠️"
                console.print(f"  {severity_icon} Validator: {suggestion['message']}")
            
            # Show warnings
            for warning in validation_result['warnings']:
                console.print(f"  ⚠️  Warning: {warning['message']}")
            
            # Show recovery info if advanced parsing was used
            if advanced_parsing and hasattr(parser, 'use_advanced_parsing') and parser.use_advanced_parsing:
                console.print(f"\n[bold yellow]🛡️  Advanced parsing recovered {len(program.statements)} statements[/bold yellow]")
            
            sys.exit(1)
        else:
            console.print("[bold green]✅ Syntax is valid![/bold green]")
            if advanced_parsing and hasattr(parser, 'use_advanced_parsing') and parser.use_advanced_parsing:
                console.print("[bold green]🔧 Advanced multi-strategy parsing successful![/bold green]")
            
            if validation_result['warnings']:
                console.print("\n[bold yellow]ℹ️  Warnings:[/bold yellow]")
                for warning in validation_result['warnings']:
                    console.print(f"  ⚠️  {warning['message']}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.pass_context
def validate(ctx, file):
    """Validate and auto-fix Zexus syntax"""
    try:
        with open(file, 'r') as f:
            source_code = f.read()
        
        syntax_style = ctx.obj['SYNTAX_STYLE']
        validator = SyntaxValidator()
        
        # Auto-detect syntax style if needed
        if syntax_style == 'auto':
            syntax_style = validator.suggest_syntax_style(source_code)
            console.print(f"🔍 [bold blue]Detected syntax style:[/bold blue] {syntax_style}")
        
        console.print(f"📝 [bold blue]Validating with {syntax_style} syntax...[/bold blue]")
        
        # Run validation and auto-fix
        fixed_code, validation_result = validator.auto_fix(source_code, syntax_style)
        
        # Show results
        if validation_result['is_valid']:
            console.print("[bold green]✅ Code is valid![/bold green]")
        else:
            console.print(f"[bold yellow]🛠️  Applied {validation_result['applied_fixes']} fixes[/bold yellow]")
            console.print("[bold yellow]⚠️  Remaining issues:[/bold yellow]")
            for suggestion in validation_result['suggestions']:
                severity_icon = "🚫" if suggestion['severity'] == 'error' else "⚠️"
                console.print(f"  {severity_icon} Line {suggestion['line']}: {suggestion['message']}")
            
            for warning in validation_result['warnings']:
                console.print(f"  ⚠️  Warning: {warning['message']}")
        
        # Write fixed code back to file if changes were made
        if validation_result['applied_fixes'] > 0:
            with open(file, 'w') as f:
                f.write(fixed_code)
            console.print(f"💾 [bold green]Updated {file} with fixes[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.pass_context
def ast(ctx, file):
    """Show AST of a Zexus file"""
    try:
        with open(file, 'r') as f:
            source_code = f.read()
        
        syntax_style = ctx.obj['SYNTAX_STYLE']
        advanced_parsing = ctx.obj['ADVANCED_PARSING']
        validator = SyntaxValidator()
        
        # Auto-detect syntax style if needed
        if syntax_style == 'auto':
            syntax_style = validator.suggest_syntax_style(source_code)
            console.print(f"🔍 [bold blue]Detected syntax style:[/bold blue] {syntax_style}")
        
        console.print(f"🔧 [bold blue]Advanced parsing:[/bold blue] {'Enabled' if advanced_parsing else 'Disabled'}")
        
        lexer = Lexer(source_code)
        parser = Parser(lexer, syntax_style, enable_advanced_strategies=advanced_parsing)
        program = parser.parse_program()
        
        parsing_method = "Advanced Multi-Strategy" if (advanced_parsing and hasattr(parser, 'use_advanced_parsing') and parser.use_advanced_parsing) else "Traditional"
        
        console.print(Panel.fit(
            str(program),
            title=f"[bold blue]Abstract Syntax Tree ({syntax_style} syntax) - {parsing_method} Parsing[/bold blue]",
            border_style="blue"
        ))
        
        if parser.errors:
            console.print("\n[bold yellow]⚠️  Parser encountered errors but continued:[/bold yellow]")
            for error in parser.errors:
                console.print(f"  ❌ {error}")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.pass_context
def tokens(ctx, file):
    """Show tokens of a Zexus file"""
    try:
        with open(file, 'r') as f:
            source_code = f.read()
        
        syntax_style = ctx.obj['SYNTAX_STYLE']
        validator = SyntaxValidator()
        
        # Auto-detect syntax style if needed
        if syntax_style == 'auto':
            syntax_style = validator.suggest_syntax_style(source_code)
            console.print(f"🔍 [bold blue]Detected syntax style:[/bold blue] {syntax_style}")
        
        lexer = Lexer(source_code)
        
        table = Table(title=f"Tokens ({syntax_style} syntax)")
        table.add_column("Type", style="cyan")
        table.add_column("Literal", style="green")
        table.add_column("Line", style="yellow")
        table.add_column("Column", style="yellow")
        
        while True:
            token = lexer.next_token()
            if token.type == "EOF":
                break
            table.add_row(token.type, token.literal, str(token.line), str(token.column))
        
        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@cli.command()
@click.pass_context
def repl(ctx):
    """Start Zexus REPL with hybrid execution"""
    syntax_style = ctx.obj['SYNTAX_STYLE']
    advanced_parsing = ctx.obj['ADVANCED_PARSING']
    execution_mode = ctx.obj['EXECUTION_MODE']
    env = Environment()
    validator = SyntaxValidator()
    
    console.print("[bold green]Zexus Hybrid REPL v1.5.0[/bold green]")
    console.print(f"🚀 [bold blue]Execution mode:[/bold blue] {execution_mode}")
    console.print(f"📝 [bold blue]Syntax style:[/bold blue] {syntax_style}")
    console.print(f"🔧 [bold blue]Advanced parsing:[/bold blue] {'Enabled' if advanced_parsing else 'Disabled'}")
    console.print("Type 'mode <interpreter|compiler|auto>' to switch execution mode")
    console.print("Type 'stats' to see execution statistics")
    console.print("Type 'exit' to quit\n")
    
    current_mode = execution_mode
    
    while True:
        try:
            code = console.input(f"[bold blue]zexus({current_mode})> [/bold blue]")
            
            if code.strip() in ['exit', 'quit']:
                break
            elif code.strip() == 'stats':
                console.print(f"📊 Interpreter uses: {orchestrator.interpreter_used}")
                console.print(f"📊 Compiler uses: {orchestrator.compiler_used}")
                console.print(f"📊 Fallbacks: {orchestrator.fallbacks}")
                continue
            elif code.strip().startswith('mode '):
                new_mode = code.split(' ')[1]
                if new_mode in ['interpreter', 'compiler', 'auto']:
                    current_mode = new_mode
                    console.print(f"🔄 Switched to {current_mode} mode")
                else:
                    console.print("❌ Invalid mode. Use: interpreter, compiler, or auto")
                continue
            elif not code.strip():
                continue
            
            # Validate syntax in REPL
            if syntax_style != 'auto':
                validation_result = validator.validate_code(code, syntax_style)
                if not validation_result['is_valid']:
                    for suggestion in validation_result['suggestions']:
                        if suggestion['severity'] == 'error':
                            console.print(f"[red]Syntax: {suggestion['message']}[/red]")
            
            # Parse and evaluate
            lexer = Lexer(code)
            parser = Parser(lexer, syntax_style, enable_advanced_strategies=advanced_parsing)
            program = parser.parse_program()
            
            if parser.errors and any("critical" in e.lower() for e in parser.errors):
                console.print("[red]Parser error:[/red]")
                for error in parser.errors:
                    console.print(f"  ❌ {error}")
                continue
            
            # UPDATED: Use evaluate from the evaluator package
            result = evaluate(program, env, debug_mode=ctx.obj['DEBUG'])
            
            if result and hasattr(result, 'inspect') and result.inspect() != 'null':
                console.print(f"[green]{result.inspect()}[/green]")
            
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")

@cli.command()
@click.option('--mode', type=click.Choice(['interpreter', 'compiler', 'auto']),
              default='auto', help='Default execution mode for the project')
@click.pass_context
def init(ctx, mode):
    """Initialize a new Zexus project with hybrid execution support"""
    syntax_style = ctx.obj['SYNTAX_STYLE']
    project_name = click.prompt("Project name", default="my-zexus-app")
    
    project_path = Path(project_name)
    project_path.mkdir(exist_ok=True)
    
    # Create basic structure
    (project_path / "src").mkdir()
    (project_path / "tests").mkdir()
    
    # Choose template based on syntax style and execution mode
    if syntax_style == "universal":
        main_content = f'''# Welcome to Zexus! (Universal Syntax)
# Execution Mode: {mode}

let app_name = "My Zexus App"

action main() {{
    print("🚀 Hello from " + app_name)
    print("✨ Running Zexus v1.5.0 in {mode} mode")
    
    # Test some features
    let numbers = [1, 2, 3, 4, 5]
    let doubled = numbers.map(transform: it * 2)
    print("Doubled numbers: " + string(doubled))
    
    # Performance test
    let start_time = time.now()
    let sum = 0
    for each number in numbers {{
        sum = sum + number
    }}
    let end_time = time.now()
    print("Sum: " + string(sum))
    print("Calculation time: " + string(end_time - start_time) + "ms")
}}

main()
'''
    else:
        main_content = f'''# Welcome to Zexus! (Flexible Syntax)
# Execution Mode: {mode}

let app_name = "My Zexus App"

action main():
    print "🚀 Hello from " + app_name
    print "✨ Running Zexus v1.5.0 in {mode} mode"
    
    # Test some features
    let numbers = [1, 2, 3, 4, 5]
    let doubled = numbers.map(transform: it * 2)
    print "Doubled numbers: " + string(doubled)
    
    # Performance test
    let start_time = time.now()
    let sum = 0
    for each number in numbers:
        sum = sum + number
    let end_time = time.now()
    print "Sum: " + string(sum)
    print "Calculation time: " + string(end_time - start_time) + "ms"

main()
'''
    
    (project_path / "main.zx").write_text(main_content)
    
    # Create config file with hybrid settings
    config_content = f'''{{
    "name": "{project_name}",
    "version": "1.5.0",
    "type": "application",
    "entry_point": "main.zx",
    "syntax_style": "{syntax_style}",
    "execution_mode": "{mode}",
    "hybrid_compiler": true,
    "fallback_to_interpreter": true
}}
'''
    
    (project_path / "zexus.json").write_text(config_content)
    
    console.print(f"\n✅ [bold green]Project '{project_name}' created![/bold green]")
    console.print(f"📁 cd {project_name}")
    console.print("🚀 zx run main.zx")
    console.print(f"📝 [bold blue]Using {syntax_style} syntax style[/bold blue]")
    console.print(f"🚀 [bold blue]Default execution mode: {mode}[/bold blue]")

@cli.command()
@click.argument('action', type=click.Choice(['on', 'off', 'minimal', 'status']))
@click.pass_context
def debug(ctx, action):
    """Control persistent debug logging: on/off/minimal/status"""
    if action == 'status':
        console.print(f"🔍 Debug level: [bold]{config.debug_level}[/bold]")
        return
    
    if action == 'on':
        config.enable_debug('full')
        console.print("✅ Debugging enabled (full)")
        return
    
    if action == 'minimal':
        config.enable_debug('minimal')
        console.print("✅ Debugging set to minimal (errors/warnings)")
        return
    
    if action == 'off':
        config.disable_debug()
        console.print("✅ Debugging disabled")
        return

@cli.command()
@click.argument('file', type=click.Path(exists=True))
@click.option('--memory/--no-memory', default=True, help='Enable memory profiling')
@click.option('--top', default=20, help='Number of functions to show in report')
@click.option('--json-output', type=click.Path(), help='Save profile data as JSON')
@click.pass_context
def profile(ctx, file, memory, top, json_output):
    """Profile performance of a Zexus program"""
    try:
        from ..profiler import Profiler
        import json as json_lib
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] Profiler module not available: {e}")
        console.print("[yellow]The profiler requires additional dependencies.[/yellow]")
        sys.exit(1)
    
    console.print(f"⚡ [bold green]Profiling[/bold green] {file}")
    console.print(f"💾 [bold blue]Memory profiling:[/bold blue] {'Enabled' if memory else 'Disabled'}")
    
    try:
        with open(file, 'r') as f:
            source_code = f.read()
        
        syntax_style = ctx.obj['SYNTAX_STYLE']
        advanced_parsing = ctx.obj['ADVANCED_PARSING']
        validator = SyntaxValidator()
        
        # Auto-detect syntax style if needed
        if syntax_style == 'auto':
            syntax_style = validator.suggest_syntax_style(source_code)
        
        # Parse the program
        lexer = Lexer(source_code, filename=file)
        parser = Parser(lexer, syntax_style, enable_advanced_strategies=advanced_parsing)
        program = parser.parse_program()
        
        if parser.errors and any("critical" in e.lower() for e in parser.errors):
            console.print("[bold red]❌ Critical parser errors, cannot profile:[/bold red]")
            for error in parser.errors:
                console.print(f"  ❌ {error}")
            sys.exit(1)
        
        # Set up environment
        env = Environment()
        import os
        abs_file = os.path.abspath(file)
        env.set("__file__", String(abs_file))
        env.set("__FILE__", String(abs_file))
        env.set("__MODULE__", String("__main__"))
        env.set("__DIR__", String(os.path.dirname(abs_file)))
        
        # Create and start profiler
        profiler = Profiler()
        profiler.start(enable_memory=memory)
        
        try:
            # Execute with profiling
            console.print("[dim]Executing with profiling...[/dim]")
            result = evaluate(program, env, debug_mode=ctx.obj['DEBUG'])
        finally:
            # Stop profiler and get report (ensure cleanup even if evaluate fails)
            report = profiler.stop()
        
        # Print report
        profiler.print_report(report, top_n=top)
        
        # Save JSON if requested
        if json_output:
            data = report.to_dict()
            with open(json_output, 'w') as f:
                json_lib.dump(data, f, indent=2)
            console.print(f"\n💾 [bold green]Profile data saved to:[/bold green] {json_output}")
        
        # Print result if any
        if result and hasattr(result, 'inspect') and result.inspect() != 'null':
            console.print(f"\n✅ [bold green]Program Result:[/bold green] {result.inspect()}")
        
    except Exception as e:
        console.print(f"[bold red]Error during profiling:[/bold red] {str(e)}")
        if ctx.obj.get('DEBUG'):
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    cli()
