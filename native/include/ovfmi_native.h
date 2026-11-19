/* SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. */
/* SPDX-License-Identifier: Apache-2.0 */

#ifndef OVFMI_NATIVE_H
#define OVFMI_NATIVE_H

#include <stdint.h>

#if defined(_WIN32)
#define OVFMI_NATIVE_API __declspec(dllimport)
#else
#define OVFMI_NATIVE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* ovfmi_native_host;

/* Error text is thread-local and remains valid until the next native call. */
OVFMI_NATIVE_API const char* ovfmi_native_last_error(void);

/* Private execution ABI, not the public USD-space FmiHost API.
 * Supports FMI 2 Real / FMI 3 Float64 scalar Co-Simulation, archived or unpacked.
 * Paths and variable names are UTF-8. Returns NULL on creation failure;
 * other operations return 0 on success or -1 on error (see last_error).
 * Create leaves the FMU instantiated. Set start values before the first step,
 * which initializes the FMU. Step uses absolute target time in seconds.
 * FMI 2 inputs are buffered until initialization; the latest value is used.
 * Their FMU errors are reported by step and require reset.
 * get_real reads the FMU state. FMI 2 reads require initialization.
 * Reset returns to instantiated state at t=0; reapply start values afterwards.
 * Handles must not be used concurrently, after destruction, or with another
 * library build. Destroy exactly once; destroying NULL is harmless.
 */
OVFMI_NATIVE_API ovfmi_native_host ovfmi_native_create(const char* fmu_path);
OVFMI_NATIVE_API int32_t ovfmi_native_set_real(
    ovfmi_native_host host, const char* variable_name, double value);
OVFMI_NATIVE_API int32_t ovfmi_native_get_real(
    ovfmi_native_host host, const char* variable_name, double* value);
OVFMI_NATIVE_API int32_t ovfmi_native_step(
    ovfmi_native_host host, double target_time);
OVFMI_NATIVE_API int32_t ovfmi_native_reset(ovfmi_native_host host);
OVFMI_NATIVE_API void ovfmi_native_destroy(ovfmi_native_host host);

#ifdef __cplusplus
}
#endif

#endif /* OVFMI_NATIVE_H */
