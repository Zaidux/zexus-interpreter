import * as path from 'path';
import * as vscode from 'vscode';
import {
    LanguageClient,
    LanguageClientOptions,
    ServerOptions,
    TransportKind
} from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
    console.log('Zexus extension is now active');

    // Register commands
    registerCommands(context);

    // Start language server
    startLanguageServer(context);
}

export function deactivate(): Thenable<void> | undefined {
    if (!client) {
        return undefined;
    }
    return client.stop();
}

function registerCommands(context: vscode.ExtensionContext) {
    // Run Zexus file
    context.subscriptions.push(
        vscode.commands.registerCommand('zexus.run', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage('No active editor');
                return;
            }

            const document = editor.document;
            if (document.languageId !== 'zexus') {
                vscode.window.showErrorMessage('Not a Zexus file');
                return;
            }

            await document.save();

            const terminal = vscode.window.createTerminal('Zexus');
            terminal.show();
            terminal.sendText(`zx run "${document.fileName}"`);
        })
    );

    // Check syntax
    context.subscriptions.push(
        vscode.commands.registerCommand('zexus.check', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage('No active editor');
                return;
            }

            const document = editor.document;
            if (document.languageId !== 'zexus') {
                vscode.window.showErrorMessage('Not a Zexus file');
                return;
            }

            await document.save();

            const terminal = vscode.window.createTerminal('Zexus Check');
            terminal.show();
            terminal.sendText(`zx check "${document.fileName}"`);
        })
    );

    // Profile performance
    context.subscriptions.push(
        vscode.commands.registerCommand('zexus.profile', async () => {
            const editor = vscode.window.activeTextEditor;
            if (!editor) {
                vscode.window.showErrorMessage('No active editor');
                return;
            }

            const document = editor.document;
            if (document.languageId !== 'zexus') {
                vscode.window.showErrorMessage('Not a Zexus file');
                return;
            }

            await document.save();

            const terminal = vscode.window.createTerminal('Zexus Profile');
            terminal.show();
            terminal.sendText(`zx profile "${document.fileName}"`);
        })
    );

    // Restart language server
    context.subscriptions.push(
        vscode.commands.registerCommand('zexus.restartLanguageServer', async () => {
            if (client) {
                await client.stop();
                await startLanguageServer(context);
                vscode.window.showInformationMessage('Zexus Language Server restarted');
            }
        })
    );
}

function startLanguageServer(context: vscode.ExtensionContext) {
    const config = vscode.workspace.getConfiguration('zexus');
    
    if (!config.get('languageServer.enabled', true)) {
        console.log('Language server disabled in settings');
        return;
    }

    // Find the Python executable
    const pythonPath = getPythonPath();
    
    // The server is implemented in Python
    const serverModule = path.join(
        context.extensionPath,
        '..',
        'src',
        'zexus',
        'lsp',
        'server.py'
    );

    // Server options
    const serverOptions: ServerOptions = {
        command: pythonPath,
        args: [serverModule],
        transport: TransportKind.stdio
    };

    // Client options
    const clientOptions: LanguageClientOptions = {
        documentSelector: [{ scheme: 'file', language: 'zexus' }],
        synchronize: {
            fileEvents: vscode.workspace.createFileSystemWatcher('**/*.zx')
        }
    };

    // Create and start the language client
    client = new LanguageClient(
        'zexusLanguageServer',
        'Zexus Language Server',
        serverOptions,
        clientOptions
    );

    client.start();
}

function getPythonPath(): string {
    const config = vscode.workspace.getConfiguration('python');
    const pythonPath = config.get<string>('defaultInterpreterPath');
    
    if (pythonPath) {
        return pythonPath;
    }

    // Fallback to common Python paths
    if (process.platform === 'win32') {
        return 'python';
    }
    return 'python3';
}
