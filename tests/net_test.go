package tests

import (
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
		front := []neuron.Neuron{
			getNeuron(t, 3),
			getNeuron(t, 3),
		}
		back := []neuron.Neuron{
			getNeuron(t, 2),
			getNeuron(t, 2),
		}
		net, err := neuron.NewNeuralNet(sensors, actions, front, back)
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
			expected := "SENSORS: sensor 1, sensor 2, sensor 3\nACTIONS: action 1, action 2\nFRONT NEURONS:\n001G00012BVVVVGQ00002O8\n001VVVVU280000G3VVVVS20\n\nBACK NEURONS:\n001FVVVV2S00002D\n0010000143VVVVTO\n\n-----\n"
			if got := net.String(); got != expected {
				t.Fatalf("expected\n%v\ngot\n%v", expected, got)
			}
		})
	})
}

func getNeuron(t *testing.T, data interface{}) neuron.Neuron {
	neu, err := neuron.NewNeuron(data)
	if err != nil {
		t.Fatalf("error intantiating neuron: %v", err)
	}
	return neu
}
