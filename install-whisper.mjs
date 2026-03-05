import { execFile } from 'child_process';
import { existsSync, mkdirSync, rmSync } from 'fs';
import { join } from 'path';

const WHISPER_REPO = 'https://github.com/ggerganov/whisper.cpp.git';
const MODEL_URL = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin';
const BINARY_NAME = 'whisper-cli';
const MODEL_NAME = 'ggml-base.en.bin';

/**
 * Install whisper.cpp binary and model to targetDir.
 * @param {string} targetDir - e.g. ~/.epistery/whisper/
 * @param {function} onProgress - callback(message) for status updates
 * @returns {{ binaryPath, modelPath }}
 */
export async function installWhisper(targetDir, onProgress = () => {}) {
  if (!existsSync(targetDir)) {
    mkdirSync(targetDir, { recursive: true });
  }

  const buildDir = join(targetDir, '_build');
  const binaryPath = join(targetDir, BINARY_NAME);
  const modelPath = join(targetDir, MODEL_NAME);

  try {
    // Step 1: Clone whisper.cpp
    onProgress('Cloning whisper.cpp repository...');
    await run('git', ['clone', '--depth', '1', WHISPER_REPO, buildDir]);
    onProgress('Clone complete.');

    // Step 2: Build with cmake
    onProgress('Configuring build (cmake)...');
    await run('cmake', ['-B', 'build'], { cwd: buildDir });
    onProgress('Compiling whisper.cpp (this takes 2-3 minutes on ARM)...');
    await run('cmake', ['--build', 'build', '-j4'], { cwd: buildDir, timeout: 600000 });
    onProgress('Compilation complete.');

    // Step 3: Copy binary and shared libraries
    const builtBinary = join(buildDir, 'build', 'bin', BINARY_NAME);
    if (!existsSync(builtBinary)) {
      throw new Error(`Build succeeded but ${BINARY_NAME} not found at ${builtBinary}`);
    }
    onProgress('Copying binary and libraries...');
    await run('cp', [builtBinary, binaryPath]);
    await run('chmod', ['+x', binaryPath]);

    // Copy shared libraries that whisper-cli depends on
    const libDir = join(buildDir, 'build', 'src');
    const ggmlLibDir = join(buildDir, 'build', 'ggml', 'src');
    for (const dir of [libDir, ggmlLibDir]) {
      if (existsSync(dir)) {
        try {
          const { stdout } = await new Promise((resolve, reject) => {
            execFile('find', [dir, '-name', '*.so*', '-type', 'f'], (err, stdout) => {
              if (err) return reject(err);
              resolve({ stdout });
            });
          });
          for (const lib of stdout.trim().split('\n').filter(Boolean)) {
            await run('cp', ['-P', lib, targetDir]);
            onProgress(`  Copied ${lib.split('/').pop()}`);
          }
        } catch (_) {}
      }
    }
    // Also search build/lib for shared libs
    const buildLibDir = join(buildDir, 'build', 'lib');
    if (existsSync(buildLibDir)) {
      try {
        const { stdout } = await new Promise((resolve, reject) => {
          execFile('find', [buildLibDir, '-name', '*.so*', '-type', 'f'], (err, stdout) => {
            if (err) return reject(err);
            resolve({ stdout });
          });
        });
        for (const lib of stdout.trim().split('\n').filter(Boolean)) {
          await run('cp', ['-P', lib, targetDir]);
          onProgress(`  Copied ${lib.split('/').pop()}`);
        }
      } catch (_) {}
    }

    // Step 4: Download model
    onProgress('Downloading base.en model (~142MB)...');
    await run('curl', ['-L', '-o', modelPath, MODEL_URL], { timeout: 600000 });
    onProgress('Model download complete.');

    // Step 5: Cleanup build dir
    onProgress('Cleaning up build files...');
    rmSync(buildDir, { recursive: true, force: true });
    onProgress('Install complete.');

    return { binaryPath, modelPath };
  } catch (err) {
    // Cleanup on failure
    try { rmSync(buildDir, { recursive: true, force: true }); } catch (_) {}
    throw err;
  }
}

/**
 * Uninstall whisper binary and model from targetDir
 */
export function uninstallWhisper(targetDir) {
  const binaryPath = join(targetDir, BINARY_NAME);
  const modelPath = join(targetDir, MODEL_NAME);
  const buildDir = join(targetDir, '_build');

  try { rmSync(binaryPath, { force: true }); } catch (_) {}
  try { rmSync(modelPath, { force: true }); } catch (_) {}
  try { rmSync(buildDir, { recursive: true, force: true }); } catch (_) {}
}

/**
 * Check if whisper is installed in targetDir
 */
export function checkWhisperInstall(targetDir) {
  const binaryPath = join(targetDir, BINARY_NAME);
  const modelPath = join(targetDir, MODEL_NAME);
  return {
    installed: existsSync(binaryPath) && existsSync(modelPath),
    binaryPath,
    modelPath,
    building: existsSync(join(targetDir, '_build'))
  };
}

/**
 * Run a command, return a promise
 */
function run(cmd, args, opts = {}) {
  return new Promise((resolve, reject) => {
    const timeout = opts.timeout || 300000;
    const child = execFile(cmd, args, {
      cwd: opts.cwd,
      timeout,
      maxBuffer: 10 * 1024 * 1024
    }, (err, stdout, stderr) => {
      if (err) {
        console.error(`[whisper-install] ${cmd} failed:`, stderr || err.message);
        return reject(new Error(`${cmd} ${args[0] || ''} failed: ${err.message}`));
      }
      resolve(stdout);
    });
  });
}
