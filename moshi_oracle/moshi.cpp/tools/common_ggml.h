#pragma once

#include <ggml.h>
#include <ggml-backend.h>
#include <ggml-cpu.h>

struct common_ggml_t {
    ggml_backend_dev_t dev;
    ggml_backend * backend;
    ggml_backend * backend_cpu;
    size_t memory_free;
    int memory_free_mb;
};

size_t device_memory_free( ggml_backend_dev_t dev ) {
    ggml_backend_dev_props props;
    ggml_backend_dev_get_props(dev, &props);
    return props.memory_free;
}

void init_ggml( common_ggml_t & ggml, const char * device = NULL, int n_threads = 0 ) {
    bool set_threads = false;
    printf("DEBUG: init_ggml starting, device=%s\n", device ? device : "NULL"); fflush(stdout);
    ggml_backend_load_all();
    printf("DEBUG: ggml_backend_load_all done\n"); fflush(stdout);
    ggml_backend * backend, * backend_cpu;
    if ( device ) {
        backend = ggml_backend_init_by_name( device, NULL );
    } else {
        backend = ggml_backend_init_best();
    }
    printf("DEBUG: backend init done, backend=%p\n", (void*)backend); fflush(stdout);
    printf("DEBUG: backend init done, backend=%p\n", (void*)backend); fflush(stdout);
    if ( ! backend ) {
        if ( ! device ) device = "best";
        fprintf( stderr, "error: failed to initialize %s backend.\n", device );
        exit(1);
    }
    printf("DEBUG: backend is not null\n"); fflush(stdout);
    auto dev = ggml_backend_get_device( backend );
    printf("DEBUG: ggml_backend_get_device done, dev=%p\n", (void*)dev); fflush(stdout);
    if ( n_threads > 0 ) {
        auto reg = ggml_backend_dev_backend_reg( dev );
        printf("DEBUG: ggml_backend_dev_backend_reg done, reg=%p\n", (void*)reg); fflush(stdout);
        auto set_n_threads = (ggml_backend_set_n_threads_t)
            ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
        printf("DEBUG: ggml_backend_reg_get_proc_address done, set_n_threads=%p\n", (void*)set_n_threads); fflush(stdout);
        if ( set_n_threads ) {
            set_n_threads( backend, n_threads );
            set_threads = true;
            printf("DEBUG: set_n_threads done\n"); fflush(stdout);
        }
    }

    printf("DEBUG: dev_type check\n"); fflush(stdout);
    if ( ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU ) {
        backend_cpu = backend;
        printf("DEBUG: dev_type is CPU\n"); fflush(stdout);
    } else {
        printf("DEBUG: dev_type is not CPU, initializing CPU backend\n"); fflush(stdout);
        backend_cpu = ggml_backend_init_by_type( GGML_BACKEND_DEVICE_TYPE_CPU, NULL );
        printf("DEBUG: backend_cpu init done, backend_cpu=%p\n", (void*)backend_cpu); fflush(stdout);
        if ( ! backend_cpu ) {
            fprintf( stderr, "error: failed to initialize a cpu device.\n" );
            exit(1);
        }
        if ( n_threads > 0 ) {
            auto dev_cpu = ggml_backend_get_device( backend_cpu );
            auto reg_cpu = ggml_backend_dev_backend_reg( dev_cpu );
            auto set_n_threads_cpu = (ggml_backend_set_n_threads_t)
                ggml_backend_reg_get_proc_address(reg_cpu, "ggml_backend_set_n_threads");
            if ( set_n_threads_cpu ) {
                set_n_threads_cpu( backend, n_threads );
            }
        }
    }

    printf("DEBUG: device_memory_free start\n"); fflush(stdout);
    auto memory_free = device_memory_free( dev );
    printf("DEBUG: device_memory_free done, memory_free=%zu\n", memory_free); fflush(stdout);

    printf("DEBUG: calling ggml_backend_dev_name\n"); fflush(stdout);
    auto dev_name = ggml_backend_dev_name( dev );
    printf("DEBUG: ggml_backend_dev_name returned pointer=%p\n", (void*)dev_name); fflush(stdout);
    if ( dev_name ) {
        printf( "using device: \"%s\"\n", dev_name );
    } else {
        printf( "using device: NULL\n" );
    }
    fflush(stdout);
    if ( set_threads ) {
        printf( "with threads: %d\n", n_threads );
    }

    ggml.dev = dev;
    ggml.backend = backend;
    ggml.backend_cpu = backend_cpu;
    ggml.memory_free = memory_free;
    ggml.memory_free_mb = (int)( memory_free / 1024 / 1024 );
}


