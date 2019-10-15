package tests

import (
	"bytes"
	"fmt"
	"io"
	"testing"
	"time"

	"github.com/cacilhas/neuron/neuron"
)

func TestNeuron(t *testing.T) {
	t.Run("NewNeuron", func(t *testing.T) {
		t.Run("[]int", func(t *testing.T) {
			input := [...]int{-5, -4, -3, -2, -1, 0, 1, 2, 3, 4}
			neu, err := neuron.NewNeuron(input[:])
			if err != nil {
				t.Fatalf("unexpected error %v", err)
			}
			if got := neu.GetSize(); got != 10 {
				t.Fatalf("expected 10, got %v", got)
			}
			for i, expect := range input {
				if got := neu.GetGene(i); got != expect {
					t.Fatalf("expected %v, got %v", expect, got)
				}
			}
			if got := neu.Compute(0, 0, 0, 0, 0, 1, 1, 1, 1, 1); got != 10 {
				t.Fatalf("expected 10, got %v", got)
			}
			str := "005FVVVVVFVVVVVSVVVVVVFVVVVVTVVVVVVG000000000001000000G0000060000020"
			if got := neu.String(); got != str {
				t.Fatalf("expected %v, got %v", str, got)
			}
		})

		t.Run("Neuron", func(t *testing.T) {
			input, _ := neuron.NewNeuron("005FVVVVVFVVVVVSVVVVVVFVVVVVTVVVVVVG000000000001000000G0000060000020")
			neu, err := neuron.NewNeuron(input)
			if err != nil {
				t.Fatalf("unexpected error %v", err)
			}
			if !neu.Equals(input) {
				t.Fatalf("expected %v, got %v", input, neu)
			}
		})

		t.Run("[]byte", func(t *testing.T) {
			input := [...]byte{0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00}
			neu, err := neuron.NewNeuron(input[:])
			if err != nil {
				t.Fatalf("unexpected error %v", err)
			}
			if got := neu.GetSize(); got != 2 {
				t.Fatalf("expected 10, got %v", got)
			}
			for i, expect := range [...]int{0, 256} {
				if got := neu.GetGene(i); got != expect {
					t.Fatalf("expected %v, got %v", expect, got)
				}
			}
			if got := neu.Compute(123, 1); got != 256 {
				t.Fatalf("expected 256, got %v", got)
			}
			str := "0010000000000080"
			if got := neu.String(); got != str {
				t.Fatalf("expected %v, got %v", str, got)
			}
		})

		t.Run("*bytes.Buffer", func(t *testing.T) {
			input := bytes.NewBuffer([]byte{0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00})
			neu, err := neuron.NewNeuron(input)
			if err != nil {
				t.Fatalf("unexpected error %v", err)
			}
			str := "0010000000000080"
			if got := neu.String(); got != str {
				t.Fatalf("expected %v, got %v", str, got)
			}
		})

		t.Run("io.Reader", func(t *testing.T) {
			r, w := io.Pipe()
			go func() {
				fmt.Fprint(w, string([]byte{0x00, 0x02, 0xff, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x01}))
				w.Close()
			}()
			neu, err := neuron.NewNeuron(r)
			if err != nil {
				t.Fatalf("unexpected error %v", err)
			}
			if got := neu.GetSize(); got != 2 {
				t.Fatalf("expected 10, got %v", got)
			}
			for i, expect := range [...]int{-1, 1} {
				if got := neu.GetGene(i); got != expect {
					t.Fatalf("expected %v, got %v", expect, got)
				}
			}
		})
	})

	t.Run("WriteToFile", func(t *testing.T) {
		r, w := io.Pipe()
		neu, _ := neuron.NewNeuron("005FVVVVVFVVVVVSVVVVVVFVVVVVTVVVVVVG000000000001000000G0000060000020")
		buf := make([]byte, 2+4*neu.GetSize())
		go func() {
			neuron.WriteToFile(w, neu)
			w.Close()
		}()
		time.Sleep(100 * time.Millisecond)
		if sz, err := r.Read(buf); err != nil || sz != 42 {
			t.Fatalf("[size %v] unexpected error %v", sz, err)
		}
		expected := []byte{
			0x00, 0x0a,
			0xff, 0xff, 0xff, 0xfb,
			0xff, 0xff, 0xff, 0xfc,
			0xff, 0xff, 0xff, 0xfd,
			0xff, 0xff, 0xff, 0xfe,
			0xff, 0xff, 0xff, 0xff,
			0x00, 0x00, 0x00, 0x00,
			0x00, 0x00, 0x00, 0x01,
			0x00, 0x00, 0x00, 0x02,
			0x00, 0x00, 0x00, 0x03,
			0x00, 0x00, 0x00, 0x04,
		}
		if string(buf) != string(expected) {
			t.Fatalf("expected %v, got %v", expected, buf)
		}
	})

	t.Run("Compute", func(t *testing.T) {
		neu, _ := neuron.NewNeuron("005FVVVVVFVVVVVSVVVVVVFVVVVVTVVVVVVG000000000001000000G0000060000020")
		t.Run("1st parameter (-5)", func(t *testing.T) {
			if got := neu.Compute(2.5, 0, 0, 0, 0, 0, 0, 0, 0, 0); got != 0 {
				t.Fatalf("expected 0, got %v", got)
			}
		})
		t.Run("2nd parameter (-4)", func(t *testing.T) {
			if got := neu.Compute(0, 2.5, 0, 0, 0, 0, 0, 0, 0, 0); got != 0 {
				t.Fatalf("expected 0, got %v", got)
			}
		})
		t.Run("3rd parameter (-3)", func(t *testing.T) {
			if got := neu.Compute(0, 0, 2.5, 0, 0, 0, 0, 0, 0, 0); got != 0 {
				t.Fatalf("expected 0, got %v", got)
			}
		})
		t.Run("4th parameter (-2)", func(t *testing.T) {
			if got := neu.Compute(0, 0, 0, 2.5, 0, 0, 0, 0, 0, 0); got != 0 {
				t.Fatalf("expected 0, got %v", got)
			}
		})
		t.Run("5th parameter (-1)", func(t *testing.T) {
			if got := neu.Compute(0, 0, 0, 0, 2.5, 0, 0, 0, 0, 0); got != 0 {
				t.Fatalf("expected 0, got %v", got)
			}
		})
		t.Run("6th parameter (0)", func(t *testing.T) {
			if got := neu.Compute(0, 0, 0, 0, 0, 2.5, 0, 0, 0, 0); got != 0 {
				t.Fatalf("expected 0, got %v", got)
			}
		})
		t.Run("7th parameter (1)", func(t *testing.T) {
			if got := neu.Compute(0, 0, 0, 0, 0, 0, 2.5, 0, 0, 0); got != 2 {
				t.Fatalf("expected 2, got %v", got)
			}
		})
		t.Run("8th parameter (2)", func(t *testing.T) {
			if got := neu.Compute(0, 0, 0, 0, 0, 0, 0, 2.5, 0, 0); got != 5 {
				t.Fatalf("expected 5, got %v", got)
			}
		})
		t.Run("9th parameter (3)", func(t *testing.T) {
			if got := neu.Compute(0, 0, 0, 0, 0, 0, 0, 0, 2.5, 0); got != 7 {
				t.Fatalf("expected 7, got %v", got)
			}
		})
		t.Run("10th parameter (4)", func(t *testing.T) {
			if got := neu.Compute(0, 0, 0, 0, 0, 0, 0, 0, 0, 2.5); got != 10 {
				t.Fatalf("expected 10, got %v", got)
			}
		})
		t.Run("missing parameter", func(t *testing.T) {
			defer func() {
				if err := recover(); err == nil {
					t.Fatalf("expected fatal not raised")
				}
			}()
			neu.Compute(0, 0, 0, 0, 0, 0, 0, 0, 0)
		})
		t.Run("extra parameter", func(t *testing.T) {
			defer func() {
				if err := recover(); err == nil {
					t.Fatalf("expected fatal not raised")
				}
			}()
			neu.Compute(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
		})
	})
}
