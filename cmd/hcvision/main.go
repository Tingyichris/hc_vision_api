package main

import (
	"encoding/base64"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os/exec"
	"strings"
	"time"
)

type worker struct {
	input  io.WriteCloser
	output io.ReadCloser
}

func (w *worker) call(payload []byte) ([]byte, error) {
	var size [4]byte
	binary.LittleEndian.PutUint32(size[:], uint32(len(payload)))
	if _, err := w.input.Write(size[:]); err != nil {
		return nil, err
	}
	if _, err := w.input.Write(payload); err != nil {
		return nil, err
	}
	if _, err := io.ReadFull(w.output, size[:]); err != nil {
		return nil, err
	}
	result := make([]byte, binary.LittleEndian.Uint32(size[:]))
	_, err := io.ReadFull(w.output, result)
	return result, err
}

func main() {
	available := make(chan *worker, 4)
	for range cap(available) {
		process := exec.Command("/home/hc/hc_vision_api/bin/tensorrt_worker", "/home/hc/weights_fp16.engine")
		input, err := process.StdinPipe()
		if err != nil {
			panic(err)
		}
		output, err := process.StdoutPipe()
		if err != nil {
			panic(err)
		}
		if err = process.Start(); err != nil {
			panic(err)
		}
		available <- &worker{input: input, output: output}
	}
	http.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"status":"ok","backend":"go-cpp-tensorrt"}`)
	})
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", 405)
			return
		}
		body, _ := io.ReadAll(io.LimitReader(r.Body, 16<<20))
		image, err := base64.StdEncoding.DecodeString(strings.TrimSpace(string(body)))
		if err != nil {
			http.Error(w, "body must be base64 image", 400)
			return
		}
		queued := time.Now()
		var selected *worker
		select {
		case selected = <-available:
		case <-r.Context().Done():
			return
		}
		queueTime := time.Since(queued)
		started := time.Now()
		result, err := selected.call(image)
		workerTime := time.Since(started)
		available <- selected
		if err != nil {
			http.Error(w, err.Error(), 503)
			return
		}
		w.Header().Set("Server-Timing", fmt.Sprintf("queue;dur=%.2f, worker;dur=%.2f",
			float64(queueTime)/float64(time.Millisecond), float64(workerTime)/float64(time.Millisecond)))
		w.Header().Set("Content-Type", "application/json")
		w.Write(result)
	})
	if err := http.ListenAndServe(":7000", nil); err != nil {
		panic(err)
	}
}
