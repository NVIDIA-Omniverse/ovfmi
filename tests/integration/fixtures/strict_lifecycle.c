// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <stdbool.h>
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
enum { INSTANTIATED, INITIALIZING, STEPPING, TERMINATED };
enum { RATE, INTEGRAL, GAIN, GUESS, INITIAL_RATE, TIME, VALUE_COUNT };

typedef struct {
    double values[VALUE_COUNT];
    int state;
} Model;

static int reset(Component component) {
    Model* model = (Model*)component;
    if (!model) return ERROR;
    *model = (Model){0};
    model->values[RATE] = 1.0;
    model->values[GAIN] = 1.0;
    return OK;
}

static Component instantiate(void) {
    Model* model = (Model*)malloc(sizeof(Model));
    reset(model);
    return model;
}

static int enter_initialization(Component component) {
    Model* model = (Model*)component;
    if (!model || model->state != INSTANTIATED) return ERROR;
    model->state = INITIALIZING;
    return OK;
}

static int exit_initialization(Component component) {
    Model* model = (Model*)component;
    if (!model || model->state != INITIALIZING) return ERROR;
    model->values[INITIAL_RATE] = model->values[RATE];
    model->state = STEPPING;
    return OK;
}

static int terminate(Component component) {
    Model* model = (Model*)component;
    if (!model || model->state != STEPPING) return ERROR;
    model->state = TERMINATED;
    return OK;
}

static int set_real(Component component, const ValueReference* refs, size_t count, const double* values) {
    Model* model = (Model*)component;
    if (!model || !refs || !values) return ERROR;
    for (size_t i = 0; i < count; ++i) {
        switch (refs[i]) {
        case RATE:
            if (values[i] < 0) return ERROR;
#if FMI_VERSION == 3
            if (model->state == INSTANTIATED) break;
#endif
            if (model->state != INITIALIZING && model->state != STEPPING) return ERROR;
            break;
        case INTEGRAL:
        case GAIN:
            if (model->state != INSTANTIATED && model->state != INITIALIZING) return ERROR;
            break;
        case GUESS:
            // initial="approx" can be set only in instantiated state.
            if (model->state != INSTANTIATED) return ERROR;
            break;
        default:
            return ERROR;
        }
        model->values[refs[i]] = values[i];
    }
    return OK;
}

static int get_real(Component component, const ValueReference* refs, size_t count, double* values) {
    Model* model = (Model*)component;
    if (!model || !refs || !values) return ERROR;
#if FMI_VERSION == 2
    if (model->state == INSTANTIATED) return ERROR;
#endif
    for (size_t i = 0; i < count; ++i) {
        if (refs[i] >= VALUE_COUNT) return ERROR;
        values[i] = model->values[refs[i]];
    }
    return OK;
}

static int do_step(Component component, double current, double step) {
    Model* model = (Model*)component;
    if (!model || model->state != STEPPING || current != model->values[TIME] || step <= 0) return ERROR;
    model->values[INTEGRAL] += model->values[GAIN] * model->values[RATE] * step;
    model->values[TIME] = current + step;
    return OK;
}

FMI_EXPORT String fmi2GetTypesPlatform(void) { return "default"; }
FMI_EXPORT String fmi2GetVersion(void) { return "2.0"; }
FMI_EXPORT Component fmi2Instantiate(String name, int type, String guid, String resources,
                                    const void* callbacks, int visible, int logging) {
    return instantiate();
}
FMI_EXPORT void fmi2FreeInstance(Component c) { free(c); }
FMI_EXPORT int fmi2SetDebugLogging(Component c, int enabled, size_t count, const String* categories) {
    return c ? OK : ERROR;
}
FMI_EXPORT int fmi2SetupExperiment(Component c, int tolerance_defined, double tolerance,
                                  double start, int stop_defined, double stop) {
    Model* model = (Model*)c;
    if (!model || model->state != INSTANTIATED) return ERROR;
    model->values[TIME] = start;
    return OK;
}
FMI_EXPORT int fmi2EnterInitializationMode(Component c) { return enter_initialization(c); }
FMI_EXPORT int fmi2ExitInitializationMode(Component c) { return exit_initialization(c); }
FMI_EXPORT int fmi2Terminate(Component c) { return terminate(c); }
FMI_EXPORT int fmi2Reset(Component c) { return reset(c); }
FMI_EXPORT int fmi2SetReal(Component c, const ValueReference* refs, size_t count, const double* values) {
    return set_real(c, refs, count, values);
}
FMI_EXPORT int fmi2GetReal(Component c, const ValueReference* refs, size_t count, double* values) {
    return get_real(c, refs, count, values);
}
FMI_EXPORT int fmi2DoStep(Component c, double current, double step, int no_rollback) {
    return do_step(c, current, step);
}

FMI_EXPORT String fmi3GetVersion(void) { return "3.0"; }
FMI_EXPORT Component fmi3InstantiateCoSimulation(
    String name, String token, String resources, bool visible, bool logging,
    bool event_mode, bool early_return, const ValueReference* required, size_t count,
    void* environment, void* logger, void* intermediate_update) {
    return instantiate();
}
FMI_EXPORT void fmi3FreeInstance(Component c) { free(c); }
FMI_EXPORT int fmi3SetDebugLogging(Component c, bool enabled, size_t count, const String* categories) {
    return c ? OK : ERROR;
}
FMI_EXPORT int fmi3EnterInitializationMode(Component c, bool tolerance_defined, double tolerance,
                                          double start, bool stop_defined, double stop) {
    int status = enter_initialization(c);
    if (status == OK) ((Model*)c)->values[TIME] = start;
    return status;
}
FMI_EXPORT int fmi3ExitInitializationMode(Component c) { return exit_initialization(c); }
FMI_EXPORT int fmi3Terminate(Component c) { return terminate(c); }
FMI_EXPORT int fmi3Reset(Component c) { return reset(c); }
FMI_EXPORT int fmi3SetFloat64(Component c, const ValueReference* refs, size_t count,
                             const double* values, size_t value_count) {
    return value_count == count ? set_real(c, refs, count, values) : ERROR;
}
FMI_EXPORT int fmi3GetFloat64(Component c, const ValueReference* refs, size_t count,
                             double* values, size_t value_count) {
    return value_count == count ? get_real(c, refs, count, values) : ERROR;
}
FMI_EXPORT int fmi3DoStep(Component c, double current, double step, bool no_rollback,
                         bool* event, bool* terminated, bool* early, double* last) {
    *event = *terminated = *early = false;
    *last = current + step;
    return do_step(c, current, step);
}

// fmi-rs loads unused functions when it loads the Co-Simulation library.
#define ERROR_STUB(name) FMI_EXPORT int name(Component c, ...) { return ERROR; }
ERROR_STUB(fmi2GetInteger)
ERROR_STUB(fmi2SetInteger)
ERROR_STUB(fmi2GetBoolean)
ERROR_STUB(fmi2SetBoolean)
ERROR_STUB(fmi2GetString)
ERROR_STUB(fmi2SetString)
ERROR_STUB(fmi2GetFMUstate)
ERROR_STUB(fmi2SetFMUstate)
ERROR_STUB(fmi2FreeFMUstate)
ERROR_STUB(fmi2SerializedFMUstateSize)
ERROR_STUB(fmi2SerializeFMUstate)
ERROR_STUB(fmi2DeSerializeFMUstate)
ERROR_STUB(fmi2GetDirectionalDerivative)
ERROR_STUB(fmi2SetRealInputDerivatives)
ERROR_STUB(fmi2GetRealOutputDerivatives)
ERROR_STUB(fmi2CancelStep)
ERROR_STUB(fmi2GetStatus)
ERROR_STUB(fmi2GetRealStatus)
ERROR_STUB(fmi2GetIntegerStatus)
ERROR_STUB(fmi2GetBooleanStatus)
ERROR_STUB(fmi2GetStringStatus)
ERROR_STUB(fmi3InstantiateModelExchange)
ERROR_STUB(fmi3InstantiateScheduledExecution)
ERROR_STUB(fmi3EnterEventMode)
ERROR_STUB(fmi3EnterStepMode)
ERROR_STUB(fmi3GetFloat32)
ERROR_STUB(fmi3SetFloat32)
ERROR_STUB(fmi3GetInt8)
ERROR_STUB(fmi3SetInt8)
ERROR_STUB(fmi3GetUInt8)
ERROR_STUB(fmi3SetUInt8)
ERROR_STUB(fmi3GetInt16)
ERROR_STUB(fmi3SetInt16)
ERROR_STUB(fmi3GetUInt16)
ERROR_STUB(fmi3SetUInt16)
ERROR_STUB(fmi3GetInt32)
ERROR_STUB(fmi3SetInt32)
ERROR_STUB(fmi3GetUInt32)
ERROR_STUB(fmi3SetUInt32)
ERROR_STUB(fmi3GetInt64)
ERROR_STUB(fmi3SetInt64)
ERROR_STUB(fmi3GetUInt64)
ERROR_STUB(fmi3SetUInt64)
ERROR_STUB(fmi3GetBoolean)
ERROR_STUB(fmi3SetBoolean)
ERROR_STUB(fmi3GetString)
ERROR_STUB(fmi3SetString)
ERROR_STUB(fmi3GetBinary)
ERROR_STUB(fmi3SetBinary)
ERROR_STUB(fmi3GetClock)
ERROR_STUB(fmi3SetClock)
ERROR_STUB(fmi3GetNumberOfVariableDependencies)
ERROR_STUB(fmi3GetVariableDependencies)
ERROR_STUB(fmi3GetFMUState)
ERROR_STUB(fmi3SetFMUState)
ERROR_STUB(fmi3FreeFMUState)
ERROR_STUB(fmi3SerializedFMUStateSize)
ERROR_STUB(fmi3SerializeFMUState)
ERROR_STUB(fmi3DeserializeFMUState)
ERROR_STUB(fmi3GetDirectionalDerivative)
ERROR_STUB(fmi3GetAdjointDerivative)
ERROR_STUB(fmi3EnterConfigurationMode)
ERROR_STUB(fmi3ExitConfigurationMode)
ERROR_STUB(fmi3GetIntervalDecimal)
ERROR_STUB(fmi3GetIntervalFraction)
ERROR_STUB(fmi3GetShiftDecimal)
ERROR_STUB(fmi3GetShiftFraction)
ERROR_STUB(fmi3SetIntervalDecimal)
ERROR_STUB(fmi3SetIntervalFraction)
ERROR_STUB(fmi3SetShiftDecimal)
ERROR_STUB(fmi3SetShiftFraction)
ERROR_STUB(fmi3EvaluateDiscreteStates)
ERROR_STUB(fmi3UpdateDiscreteStates)
ERROR_STUB(fmi3EnterContinuousTimeMode)
ERROR_STUB(fmi3CompletedIntegratorStep)
ERROR_STUB(fmi3SetTime)
ERROR_STUB(fmi3SetContinuousStates)
ERROR_STUB(fmi3GetContinuousStateDerivatives)
ERROR_STUB(fmi3GetEventIndicators)
ERROR_STUB(fmi3GetContinuousStates)
ERROR_STUB(fmi3GetNominalsOfContinuousStates)
ERROR_STUB(fmi3GetNumberOfEventIndicators)
ERROR_STUB(fmi3GetNumberOfContinuousStates)
ERROR_STUB(fmi3GetOutputDerivatives)
ERROR_STUB(fmi3ActivateModelPartition)
#undef ERROR_STUB
#undef FMI_EXPORT
