/* SiLVANN.exe — the portable app's launcher (NN-162).
 *
 * The folder unzipped anywhere: SiLVANN.exe beside `python\` (Python's embeddable build with the runtime's packages),
 * `lib\` (the evaluator and the silicon families), the release's tree and `scripts\portable\silvann_portable.py`. A
 * double-click starts the app — the UI in a window of its own (▶ silvann_portable.py) — inside a Windows job object that
 * kills every process in it when the job's last handle closes: this launcher holds it, so the app's end, a crash, or this
 * process ended from the Task Manager ends the UI, every model server it started and their children with it. No console,
 * no installer, no administrator.
 *
 * Built by scripts\portable\build_portable.ps1 (MSVC: cl /O2 SiLVANN.c /link /SUBSYSTEM:WINDOWS user32.lib). */
#define WIN32_LEAN_AND_MEAN
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <wchar.h>

static void say(const wchar_t* what) {
    MessageBoxW(NULL, what, L"SiLVANN", MB_OK | MB_ICONERROR);
}

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR command, int show) {
    (void)instance; (void)previous; (void)command; (void)show;
    wchar_t dir[MAX_PATH], py[MAX_PATH], app[MAX_PATH], line[3 * MAX_PATH];
    const DWORD n = GetModuleFileNameW(NULL, dir, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) { say(L"SiLVANN could not read where it is."); return 1; }
    wchar_t* slash = wcsrchr(dir, L'\\');
    if (slash != NULL) *slash = L'\0';
    swprintf(py, MAX_PATH, L"%ls\\python\\pythonw.exe", dir);
    swprintf(app, MAX_PATH, L"%ls\\scripts\\portable\\silvann_portable.py", dir);
    if (GetFileAttributesW(py) == INVALID_FILE_ATTRIBUTES || GetFileAttributesW(app) == INVALID_FILE_ATTRIBUTES) {
        say(L"This folder is not SiLVANN's whole: python\\pythonw.exe or scripts\\portable\\silvann_portable.py is missing. "
            L"Unzip silvann_portable_windows_x64.zip again, whole.");
        return 1;
    }
    swprintf(line, 3 * MAX_PATH, L"\"%ls\" \"%ls\"", py, app);

    /* the job: every process in it killed when its last handle — this process's — closes */
    HANDLE job = CreateJobObjectW(NULL, NULL);
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits;
    ZeroMemory(&limits, sizeof limits);
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if (job == NULL || !SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof limits)) {
        say(L"SiLVANN could not make the job its processes run in.");
        return 1;
    }

    /* the app, started suspended, put in the job before it runs a line, so everything it starts is in it too */
    STARTUPINFOW si;
    PROCESS_INFORMATION pi;
    ZeroMemory(&si, sizeof si);
    si.cb = sizeof si;
    ZeroMemory(&pi, sizeof pi);
    if (!CreateProcessW(py, line, NULL, NULL, FALSE, CREATE_SUSPENDED | CREATE_NO_WINDOW, NULL, dir, &si, &pi)) {
        say(L"SiLVANN could not start its Python (python\\pythonw.exe).");
        return 1;
    }
    if (!AssignProcessToJobObject(job, pi.hProcess)) {
        TerminateProcess(pi.hProcess, 1);
        say(L"SiLVANN could not put its app in its job.");
        return 1;
    }
    ResumeThread(pi.hThread);
    CloseHandle(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 0;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hProcess);
    CloseHandle(job);                   /* the job's last handle: whatever the app left running ends here */
    return (int)code;
}
