# ~/zexus-interpreter/src/zexus/cli/main.py
import click
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.table import Table

# Import your existing modules
from ..lexer import Lexer
from ..parser import Parser
from ..evaluator import eval_node, Environment

console = Console()

@click.group()
@click.version_option(version="0.1.0", prog_name="Zexus")
def cli():
    """Zexus Programming Language - Declarative, intent-based programming"""
    pass

@cli.command()
@click.argument('file', type=click.Path(exists=True))
def run(file):
    """Run a Zexus program"""
    try:
        with open(file, 'r') as f:
            source_code = f.read()
        
        console.print(f"🚀 [bold green]Running[/bold green] {file}\n")
        
        # Run the program using your existing code
        lexer = Lexer(source_code)
        parser = Parser(lexer)
        program = parser.parse_program()
        
        if parser.errors:
            console.print("[bold red]Parser Errors:[/bold red]")
            for error in parser.errors:
                console.print(f"  ❌ {error}")
            return
        
        env = Environment()
        result = eval_node(program, env)
        
        if result and hasattr(result, 'inspect') and result.inspect() != 'null':
            console.print(f"\n✅ [bold green]Result:[/bold green] {result.inspect()}")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)

@cli.command()
@click.argument('file', type=click.Path(exists=True))
def check(file):
    """Check syntax of a Zexus file"""
    try:
        with open(file, 'r') as f:
            source_code = f.read()
        
        lexer = Lexer(source_code)
        parser = Parser(lexer)
        program = parser.parse_program()
        
        if parser.errors:
            console.print("[bold red]❌ Syntax Errors Found:[/bold red]")
            for error in parser.errors:
                console.print(f"  {error}")
            sys.exit(1)
        else:
            console.print("[bold green]✅ Syntax is valid![/bold green]")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)

@cli.command()
@click.argument('file', type=click.Path(exists=True))
def ast(file):
    """Show AST of a Zexus file"""
    try:
        with open(file, 'r') as f:
            source_code = f.read()
        
        lexer = Lexer(source_code)
        parser = Parser(lexer)
        program = parser.parse_program()
        
        console.print(Panel.fit(
            str(program),
            title="[bold blue]Abstract Syntax Tree[/bold blue]",
            border_style="blue"
        ))
        
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")

@cli.command()
@click.argument('file', type=click.Path(exists=True))
def tokens(file):
    """Show tokens of a Zexus file"""
    try:
        with open(file, 'r') as f:
            source_code = f.read()
        
        lexer = Lexer(source_code)
        
        table = Table(title="Tokens")
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
def repl():
    """Start Zexus REPL"""
    env = Environment()
    console.print("[bold green]Zexus REPL v0.1.0[/bold green]")
    console.print("Type 'exit' to quit\n")
    
    while True:
        try:
            code = console.input("[bold blue]>>> [/bold blue]")
            if code.strip() in ['exit', 'quit']:
                break
                
            if not code.strip():
                continue
                
            lexer = Lexer(code)
            parser = Parser(lexer)
            program = parser.parse_program()
            
            if parser.errors:
                for error in parser.errors:
                    console.print(f"[red]Error: {error}[/red]")
                continue
                
            result = eval_node(program, env)
            if result and hasattr(result, 'inspect') and result.inspect() != 'null':
                console.print(f"[green]{result.inspect()}[/green]")
                
        except KeyboardInterrupt:
            console.print("\n👋 Goodbye!")
            break
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")

@cli.command()
def init():
    """Initialize a new Zexus project"""
    project_name = click.prompt("Project name", default="my-zexus-app")
    
    project_path = Path(project_name)
    project_path.mkdir(exist_ok=True)
    
    # Create basic structure
    (project_path / "src").mkdir()
    (project_path / "tests").mkdir()
    
    # Create main file
    main_content = '''# Welcome to Zexus!

let app_name = "My Zexus App"

action main():
    print "🚀 Hello from " + app_name
    print "✨ Running Zexus v0.1.0"
    
    # Test some features
    let numbers = [1, 2, 3, 4, 5]
    let doubled = numbers.map(transform: it * 2)
    print "Doubled numbers: " + string(doubled)

main()
'''
    
    (project_path / "main.zx").write_text(main_content)
    
    # Create utils file
    utils_content = '''# Utility functions

action add(a: integer, b: integer) -> integer:
    return a + b

action multiply(a: integer, b: integer) -> integer:
    return a * b

action greet(name: text) -> text:
    return "Hello, " + name + "!"
'''
    
    (project_path / "src" / "utils.zx").write_text(utils_content)
    
    # Create config file
    config_content = '''{
    "name": "''' + project_name + '''",
    "version": "0.1.0",
    "type": "application",
    "entry_point": "main.zx"
}
'''
    
    (project_path / "zexus.json").write_text(config_content)
    
    console.print(f"\n✅ [bold green]Project '{project_name}' created![/bold green]")
    console.print(f"📁 cd {project_name}")
    console.print("🚀 zx run main.zx")

if __name__ == "__main__":
    cli()
