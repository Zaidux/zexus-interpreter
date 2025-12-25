#!/usr/bin/env python3
"""
ZPM - Zexus Package Manager CLI
"""
import click
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zexus.zpm import PackageManager

console = Console()


@click.group()
@click.version_option(version="1.5.0", prog_name="ZPM")
def cli():
    """ZPM - Zexus Package Manager
    
    Manage packages for your Zexus projects.
    """
    pass


@cli.command()
@click.option('--name', '-n', help='Package name')
@click.option('--version', '-v', default='1.5.0', help='Initial version')
def init(name, version):
    """Initialize a new Zexus project"""
    pm = PackageManager()
    config = pm.init(name=name, version=version)
    
    console.print(Panel.fit(
        f"[bold green]Project initialized![/bold green]\n\n"
        f"📦 Name: {config['name']}\n"
        f"🏷️  Version: {config['version']}\n"
        f"📄 Config: zexus.json",
        title="ZPM",
        border_style="green"
    ))


@cli.command()
@click.argument('package', required=False)
@click.option('--dev', '-D', is_flag=True, help='Save to devDependencies')
@click.option('--save', '-S', is_flag=True, default=True, help='Save to dependencies')
def install(package, dev, save):
    """Install a package or all packages from zexus.json
    
    Examples:
        zpm install              # Install all dependencies
        zpm install std          # Install std package
        zpm install web@0.2.0    # Install specific version
        zpm install crypto -D    # Install as dev dependency
    """
    pm = PackageManager()
    
    if package:
        success = pm.install(package, dev=dev)
    else:
        success = pm.install_all()
    
    if not success:
        sys.exit(1)


@cli.command()
@click.argument('package')
def uninstall(package):
    """Uninstall a package
    
    Example:
        zpm uninstall std
    """
    pm = PackageManager()
    success = pm.uninstall(package)
    
    if not success:
        sys.exit(1)


@cli.command(name='list')
def list_packages():
    """List installed packages"""
    pm = PackageManager()
    packages = pm.list()
    
    if not packages:
        console.print("📦 No packages installed")
        return
    
    table = Table(title=f"Installed Packages ({len(packages)})")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Path", style="dim")
    
    for pkg in packages:
        table.add_row(pkg["name"], pkg["version"], pkg["path"])
    
    console.print(table)


@cli.command()
@click.argument('query')
def search(query):
    """Search for packages
    
    Example:
        zpm search crypto
    """
    pm = PackageManager()
    results = pm.search(query)
    
    if not results:
        console.print(f"🔍 No packages found for '{query}'")
        return
    
    table = Table(title=f"Search Results for '{query}'")
    table.add_column("Name", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Description", style="white")
    
    for pkg in results:
        table.add_row(
            pkg["name"],
            pkg["version"],
            pkg.get("description", "")
        )
    
    console.print(table)


@cli.command()
def publish():
    """Publish package to registry
    
    Publishes the current package to the ZPM registry.
    Requires a valid zexus.json with name and version.
    """
    pm = PackageManager()
    success = pm.publish()
    
    if not success:
        sys.exit(1)


@cli.command()
def info():
    """Show project information"""
    pm = PackageManager()
    config = pm.load_config()
    
    if not config:
        console.print("❌ No zexus.json found. Run 'zpm init' first.")
        sys.exit(1)
    
    console.print(Panel.fit(
        f"[bold cyan]Project Information[/bold cyan]\n\n"
        f"📦 Name: {config.get('name', 'N/A')}\n"
        f"🏷️  Version: {config.get('version', 'N/A')}\n"
        f"📝 Description: {config.get('description', 'N/A')}\n"
        f"👤 Author: {config.get('author', 'N/A')}\n"
        f"📜 License: {config.get('license', 'N/A')}\n"
        f"📄 Main: {config.get('main', 'N/A')}",
        title="ZPM Info",
        border_style="cyan"
    ))
    
    # Show dependencies
    deps = config.get('dependencies', {})
    if deps:
        console.print("\n[bold]Dependencies:[/bold]")
        for name, version in deps.items():
            console.print(f"  • {name}@{version}")
    
    dev_deps = config.get('devDependencies', {})
    if dev_deps:
        console.print("\n[bold]Dev Dependencies:[/bold]")
        for name, version in dev_deps.items():
            console.print(f"  • {name}@{version}")


@cli.command()
def clean():
    """Remove zpm_modules directory"""
    pm = PackageManager()
    
    if not pm.zpm_dir.exists():
        console.print("✅ Already clean (no zpm_modules)")
        return
    
    import shutil
    shutil.rmtree(pm.zpm_dir)
    console.print(f"✅ Removed {pm.zpm_dir}")


if __name__ == '__main__':
    cli()
