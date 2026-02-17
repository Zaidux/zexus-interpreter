/**
 * Zexus Debug Adapter — thin shim that spawns the Python DAP server
 * and proxies DAP messages between VS Code and the Python process.
 *
 * VS Code ↔ (stdin/stdout) ↔ debugAdapter.ts ↔ (stdin/stdout) ↔ python -m zexus.dap
 */

import * as cp from 'child_process';
import * as path from 'path';

// VS Code communicates with this process over stdin/stdout using DAP.
// We simply forward everything to the Python DAP server process.

const pythonPath = process.env.ZEXUS_PYTHON || 'python3';

const server = cp.spawn(pythonPath, ['-m', 'zexus.dap'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    cwd: process.cwd(),
});

// Forward VS Code stdin → Python stdin
process.stdin.resume();
process.stdin.on('data', (data: Buffer) => {
    if (server.stdin && !server.stdin.destroyed) {
        server.stdin.write(data);
    }
});

// Forward Python stdout → VS Code stdout
server.stdout.on('data', (data: Buffer) => {
    process.stdout.write(data);
});

// Forward Python stderr → VS Code stderr (for diagnostics)
server.stderr.on('data', (data: Buffer) => {
    process.stderr.write(data);
});

server.on('exit', (code: number | null) => {
    process.exit(code ?? 0);
});

process.on('exit', () => {
    server.kill();
});
