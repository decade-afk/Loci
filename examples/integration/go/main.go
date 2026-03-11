package main

/*
#cgo CFLAGS: -I../../../include
#cgo windows LDFLAGS: -L../../../target/release -lloci
#cgo linux LDFLAGS: -L../../../target/release -lloci -ldl -lm -lpthread
#cgo darwin LDFLAGS: -L../../../target/release -lloci
#include "loci.h"
#include <stdlib.h>
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func main() {
	model := C.CString("D:/OpenProject/Qwen_Qwen3-0.6B-Q5_K_L.gguf")
	defer C.free(unsafe.Pointer(model))

	engine := C.loci_engine_new(model, 512, 0)
	if engine == nil {
		err := C.loci_get_last_error()
		if err != nil {
			fmt.Printf("loci_engine_new failed: %s\n", C.GoString(err))
		} else {
			fmt.Println("loci_engine_new failed: (no error)")
		}
		return
	}
	defer C.loci_engine_free(engine)

	prompt := C.CString("Hello from Go")
	defer C.free(unsafe.Pointer(prompt))

	out := C.loci_generate(engine, prompt, 32, C.float(0.7))
	if out == nil {
		err := C.loci_get_last_error()
		if err != nil {
			fmt.Printf("loci_generate failed: %s\n", C.GoString(err))
		} else {
			fmt.Println("loci_generate failed: (no error)")
		}
		return
	}
	defer C.loci_free_string(out)

	fmt.Println(C.GoString(out))
}
