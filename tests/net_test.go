package tests

import (
	"bytes"
	"encoding/base32"
	"io"
	"math/rand"
	"strings"
	"testing"

	"github.com/cacilhas/neuron/neuron"
)

func TestNet(t *testing.T) {
	rand.Seed(0)

	t.Run("NewNeuralNet", func(t *testing.T) {
		sensors := []string{"sensor 3", "sensor 2", "sensor 1"}
		actions := []string{"action 2", "action 1"}
		front := neuron.Layer{
			getNeuron(t, 3),
			getNeuron(t, 3),
		}
		back := neuron.Layer{
			getNeuron(t, 2),
			getNeuron(t, 2),
		}
		net, err := neuron.NewNeuralNet(sensors, actions, []neuron.Layer{front, back})
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}

		t.Run("GetActions", func(t *testing.T) {
			expected := "action 1, action 2"
			if got := strings.Join(net.GetActions(), ", "); got != expected {
				t.Fatalf("expected %v, got %v", expected, got)
			}
		})

		t.Run("GetSensors", func(t *testing.T) {
			expected := "sensor 1, sensor 2, sensor 3"
			if got := strings.Join(net.GetSensors(), ", "); got != expected {
				t.Fatalf("expected %v, got %v", expected, got)
			}
		})

		t.Run("Compute", func(t *testing.T) {
			t.Run("1st", func(t *testing.T) {
				params := map[string]float64{
					"sensor 1": 1,
					"sensor 2": 1,
					"sensor 3": 1,
				}
				got, err := net.Compute(params)
				if err != nil {
					t.Fatalf("unexpected error %v", err)
				}
				if got["action 1"] {
					t.Fatalf("action 1 expected not to be trigged")
				}
				if !got["action 2"] {
					t.Fatalf("action 2 expected to be trigged")
				}
			})
			t.Run("2nd", func(t *testing.T) {
				params := map[string]float64{
					"sensor 1": 10,
					"sensor 2": 2.5,
					"sensor 3": -100,
				}
				got, err := net.Compute(params)
				if err != nil {
					t.Fatalf("unexpected error %v", err)
				}
				if !got["action 1"] {
					t.Fatalf("action 1 expected to be trigged")
				}
				if got["action 2"] {
					t.Fatalf("action 2 expected not to be trigged")
				}
			})
		})

		t.Run("String", func(t *testing.T) {
			expected := "SENSORS: sensor 1, sensor 2, sensor 3\nACTIONS: action 1, action 2\nNEURONS:\n001G00012BVVVVGQ00002O8\n001VVVVU280000G3VVVVS20\n\n001FVVVV2S00002D\n0010000143VVVVTO\n\n-----\n"
			if got := net.String(); got != expected {
				t.Fatalf("expected\n%v\ngot\n%v", expected, got)
			}
		})

		t.Run("Save", func(t *testing.T) {
			r, w := io.Pipe()
			go func() {
				net.Save(w)
				w.Close()
			}()
			var buf bytes.Buffer
			buf.ReadFrom(r)
			got := base32.HexEncoding.WithPadding(base32.NoPadding).EncodeToString(buf.Bytes())
			expected := "01QG00000C000SR5DPPMUSH064076PBEEDNN481I01PMARJJDTP20CO000100031CDQ6IRRE40OG0OB3EHKMURH0680000G000004000001G00012BVVVVGQ00002O800FVVVVGI000040VVVVV0G002000000NVVVVHE000016G00G0000I1VVVVUS0000000"
			if got != expected {
				t.Fatalf("expected %v, got %v", expected, got)
			}
		})
	})

	t.Run("LoadNet", func(t *testing.T) {
		data, _ := base32.HexEncoding.WithPadding(base32.NoPadding).DecodeString("01QG00000C000SR5DPPMUSH064076PBEEDNN481I01PMARJJDTP20CO000100031CDQ6IRRE40OG0OB3EHKMURH0680000G000004000001G00012BVVVVGQ00002O800FVVVVGI000040VVVVV0G002000000NVVVVHE000016G00G0000I1VVVVUS0000000")
		r, w := io.Pipe()
		go func() {
			w.Write(data)
			w.Close()
		}()
		net, err := neuron.LoadNet(r)
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}

		t.Run("GetActions", func(t *testing.T) {
			expected := "action 1, action 2"
			if got := strings.Join(net.GetActions(), ", "); got != expected {
				t.Fatalf("expected %v, got %v", expected, got)
			}
		})

		t.Run("GetSensors", func(t *testing.T) {
			expected := "sensor 1, sensor 2, sensor 3"
			if got := strings.Join(net.GetSensors(), ", "); got != expected {
				t.Fatalf("expected %v, got %v", expected, got)
			}
		})

		t.Run("GetFrontNeurons", func(t *testing.T) {
			var buf strings.Builder
			for _, neu := range net.GetNeurons(0) {
				buf.WriteString(neu.String())
				buf.WriteByte(0x20)
			}
			expected := "001G00012BVVVVGQ00002O8 001VVVVU280000G3VVVVS20 "
			if got := buf.String(); got != expected {
				t.Fatalf("expected %v, got %v", expected, got)
			}
		})

		t.Run("GetBackNeurons", func(t *testing.T) {
			var buf strings.Builder
			for _, neu := range net.GetNeurons(1) {
				buf.WriteString(neu.String())
				buf.WriteByte(0x20)
			}
			expected := "001FVVVV2S00002D 0010000143VVVVTO "
			if got := buf.String(); got != expected {
				t.Fatalf("expected %v, got %v", expected, got)
			}
		})

		t.Run("GetChild", func(t *testing.T) {
			child := net.GetChild(100)
			expected := "SENSORS: sensor 1, sensor 2, sensor 3\nACTIONS: action 1, action 2\nNEURONS:\n001G00014JVVVVGN00002QG\n001VVVVU200000G5VVVVRS0\n\n001FVVVUU000002D\n0010000143VVVVV4\n\n-----\n"
			if got := child.String(); got != expected {
				t.Fatalf("expected:\n%v\ngot:%v", expected, got)
			}
		})
	})
}

func getNeuron(t *testing.T, data int) neuron.Neuron {
	neu, err := neuron.NewNeuron(data)
	if err != nil {
		t.Fatalf("error intantiating neuron: %v", err)
	}
	return neu
}
