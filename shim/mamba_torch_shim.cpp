// Compatibility shim for mamba-ssm / causal-conv1d compiled against torch <2.11
// Uses dlsym to resolve new symbols at call-time — no torch link dependency.

#include <dlfcn.h>
#include <cstdint>

extern "C" {

// Old: c10_cuda_check_implementation(..., int line_number, ...)
// New: c10_cuda_check_implementation(..., uint32_t line_number, ...)
void _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib(
    int32_t err, const char* filename, const char* function_name,
    int line_number, bool include_device_assertions)
{
    typedef void (*Fn)(int32_t, const char*, const char*, uint32_t, bool);
    static Fn fn = reinterpret_cast<Fn>(
        dlsym(RTLD_DEFAULT, "_ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_jb"));
    if (fn) fn(err, filename, function_name,
               static_cast<uint32_t>(line_number), include_device_assertions);
}

// Old: SetDevice(signed char)
// New: SetDevice(signed char, bool force=false)
void _ZN3c104cuda9SetDeviceEa(signed char device)
{
    typedef void (*Fn)(signed char, bool);
    static Fn fn = nullptr;
    if (!fn) fn = reinterpret_cast<Fn>(
        dlsym(RTLD_DEFAULT, "_ZN3c104cuda9SetDeviceEab"));
    if (fn) fn(device, false);
}

} // extern "C"
