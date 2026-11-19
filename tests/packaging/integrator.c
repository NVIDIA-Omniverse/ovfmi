// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stddef.h>
#include <stdlib.h>

#ifdef _WIN32
#define FMI_EXPORT __declspec(dllexport)
#else
#define FMI_EXPORT __attribute__((visibility("default")))
#endif

typedef void* Component;
typedef unsigned int ValueReference;
typedef const char* String;

enum { OK = 0, ERROR = 3 };

typedef struct {
    double rate;
    double integral;
    double time;
    int initialized;
} Integrator;

FMI_EXPORT const char* fmi2GetTypesPlatform(void) { return "default"; }
FMI_EXPORT const char* fmi2GetVersion(void) { return "2.0"; }

FMI_EXPORT Component fmi2Instantiate(
    String name, int type, String guid, String resource_location,
    const void* callbacks, int visible, int logging) {
    Integrator* model = (Integrator*)calloc(1, sizeof(Integrator));
    if (model) model->rate = 1.0;
    return model;
}

FMI_EXPORT void fmi2FreeInstance(Component component) { free(component); }

FMI_EXPORT int fmi2SetDebugLogging(Component component, int enabled, size_t count, const String* categories) {
    return component ? OK : ERROR;
}

FMI_EXPORT int fmi2SetupExperiment(
    Component component, int tolerance_defined, double tolerance,
    double start, int stop_defined, double stop) {
    Integrator* model = (Integrator*)component;
    if (!model) return ERROR;
    model->time = start;
    return OK;
}

FMI_EXPORT int fmi2EnterInitializationMode(Component component) { return component ? OK : ERROR; }

FMI_EXPORT int fmi2ExitInitializationMode(Component component) {
    Integrator* model = (Integrator*)component;
    if (!model) return ERROR;
#ifdef INITIALIZE_FROM_INPUT
    model->integral = model->rate;
#endif
    model->initialized = 1;
    return OK;
}

FMI_EXPORT int fmi2Terminate(Component component) {
    Integrator* model = (Integrator*)component;
    if (!model) return ERROR;
    model->initialized = 0;
    return OK;
}

FMI_EXPORT int fmi2Reset(Component component) {
    Integrator* model = (Integrator*)component;
    if (!model) return ERROR;
    model->rate = 1.0;
    model->integral = 0.0;
    model->time = 0.0;
    model->initialized = 0;
    return OK;
}

FMI_EXPORT int fmi2DoStep(Component component, double current, double step, int no_rollback) {
    Integrator* model = (Integrator*)component;
    if (!model || !model->initialized || current != model->time || step <= 0.0) return ERROR;
    model->integral += model->rate * step;
    model->time = current + step;
    return OK;
}

FMI_EXPORT int fmi2SetReal(Component component, const ValueReference* refs, size_t count, const double* values) {
    Integrator* model = (Integrator*)component;
    if (!model || !refs || !values) return ERROR;
    for (size_t i = 0; i < count; ++i) {
        if (refs[i] != 0) return ERROR;
        model->rate = values[i];
    }
    return OK;
}

FMI_EXPORT int fmi2GetReal(Component component, const ValueReference* refs, size_t count, double* values) {
    Integrator* model = (Integrator*)component;
    if (!model || !refs || !values) return ERROR;
    for (size_t i = 0; i < count; ++i) {
        if (refs[i] == 0) values[i] = model->rate;
        else if (refs[i] == 1) values[i] = model->integral;
        else return ERROR;
    }
    return OK;
}

// fmi-rs loads the complete Co-Simulation symbol table at instantiation.
FMI_EXPORT int fmi2GetInteger(Component c, const ValueReference* r, size_t n, int* v) { return ERROR; }
FMI_EXPORT int fmi2SetInteger(Component c, const ValueReference* r, size_t n, const int* v) { return ERROR; }
FMI_EXPORT int fmi2GetBoolean(Component c, const ValueReference* r, size_t n, int* v) { return ERROR; }
FMI_EXPORT int fmi2SetBoolean(Component c, const ValueReference* r, size_t n, const int* v) { return ERROR; }
FMI_EXPORT int fmi2GetString(Component c, const ValueReference* r, size_t n, String* v) { return ERROR; }
FMI_EXPORT int fmi2SetString(Component c, const ValueReference* r, size_t n, const String* v) { return ERROR; }
FMI_EXPORT int fmi2GetFMUstate(Component c, void** state) { return ERROR; }
FMI_EXPORT int fmi2SetFMUstate(Component c, void* state) { return ERROR; }
FMI_EXPORT int fmi2FreeFMUstate(Component c, void** state) { return ERROR; }
FMI_EXPORT int fmi2SerializedFMUstateSize(Component c, void* state, size_t* size) { return ERROR; }
FMI_EXPORT int fmi2SerializeFMUstate(Component c, void* state, char* bytes, size_t size) { return ERROR; }
FMI_EXPORT int fmi2DeSerializeFMUstate(Component c, const char* bytes, size_t size, void** state) { return ERROR; }
FMI_EXPORT int fmi2GetDirectionalDerivative(Component c, const ValueReference* unknowns, size_t nu, const ValueReference* knowns, size_t nk, const double* seed, double* result) { return ERROR; }
FMI_EXPORT int fmi2SetRealInputDerivatives(Component c, const ValueReference* refs, size_t count, const int* order, const double* values) { return ERROR; }
FMI_EXPORT int fmi2GetRealOutputDerivatives(Component c, const ValueReference* refs, size_t count, const int* order, double* values) { return ERROR; }
FMI_EXPORT int fmi2CancelStep(Component c) { return ERROR; }
FMI_EXPORT int fmi2GetStatus(Component c, int kind, int* value) { return ERROR; }
FMI_EXPORT int fmi2GetRealStatus(Component c, int kind, double* value) { return ERROR; }
FMI_EXPORT int fmi2GetIntegerStatus(Component c, int kind, int* value) { return ERROR; }
FMI_EXPORT int fmi2GetBooleanStatus(Component c, int kind, int* value) { return ERROR; }
FMI_EXPORT int fmi2GetStringStatus(Component c, int kind, String* value) { return ERROR; }

#undef FMI_EXPORT
