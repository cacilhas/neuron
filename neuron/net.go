package neuron

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"sort"
	"strings"
)

// NeuralNet represents a neural net
type NeuralNet interface {
	GetActions() []string
	GetSensors() []string
	Compute(map[string]float64) (map[string]bool, error)
	Save(io.Writer) error
	String() string
}

type neuralnet struct {
	actions        []string
	frontNeuronSet []Neuron
	backNeuronSet  []Neuron
	sensors        []string
}

// NewNeuralNet instantiate a new neural net
func NewNeuralNet(sensors, actions []string, frontNeuronSet, backNeuronSet []Neuron) (NeuralNet, error) {
	sortedSensors := usort(sensors)
	sortedActions := usort(actions)
	sensorsCount := len(sortedSensors)
	actionsCount := len(sortedActions)

	if sensorsCount == 0 {
		return nil, fmt.Errorf("no sensor supplied")
	}

	if actionsCount == 0 {
		return nil, fmt.Errorf("no action supplied")
	}

	if len(frontNeuronSet) != actionsCount {
		return nil, fmt.Errorf("expected %v front neurons, got %v", actionsCount, len(frontNeuronSet))
	}

	if len(backNeuronSet) != actionsCount {
		return nil, fmt.Errorf("expected %v back neurons, got %v", actionsCount, len(backNeuronSet))
	}

	for i, neuron := range frontNeuronSet {
		if neuron.GetSize() != sensorsCount {
			return nil, fmt.Errorf("[fron neuron %v] expected %v genes, got %v", i, sensorsCount, neuron.GetSize())
		}
	}

	for i, neuron := range backNeuronSet {
		if neuron.GetSize() != actionsCount {
			return nil, fmt.Errorf("[back neuron %v] expected %v genes, got %v", i, actionsCount, neuron.GetSize())
		}
	}

	return &neuralnet{sortedActions, frontNeuronSet, backNeuronSet, sortedSensors}, nil
}

func (net neuralnet) GetActions() []string {
	actions := make([]string, len(net.actions))
	copy(actions, net.actions)
	return actions
}

func (net neuralnet) GetSensors() []string {
	sensors := make([]string, len(net.sensors))
	copy(sensors, net.sensors)
	return sensors
}

func (net neuralnet) Compute(incoming map[string]float64) (map[string]bool, error) {
	if err := net.checkInput(incoming); err != nil {
		return nil, err
	}

	middle := make([]int, len(net.actions))

	for i, neuron := range net.frontNeuronSet {
		input := make([]float64, len(incoming))
		for j, sensor := range net.sensors {
			input[j] = incoming[sensor]
		}
		middle[i] = neuron.Compute(input...)
	}

	res := make(map[string]bool)

	for i, neuron := range net.backNeuronSet {
		input := make([]float64, len(middle))
		for j, value := range middle {
			input[j] = float64(value)
		}
		res[net.actions[i]] = neuron.Compute(input...) > 0
	}

	return res, nil
}

func (net neuralnet) checkInput(incoming map[string]float64) error {
	if len(incoming) != len(net.sensors) {
		return fmt.Errorf("incoming mismatch sensors")
	}
	sensors := make(map[string]bool)
	for _, sensor := range net.sensors {
		sensors[sensor] = true
	}
	for key := range incoming {
		if !sensors[key] {
			return fmt.Errorf("incoming mismatch sensors")
		}
	}
	return nil
}

func (net neuralnet) String() string {
	var buf strings.Builder
	buf.WriteString("SENSORS: ")
	buf.WriteString(strings.Join(net.sensors, ", "))
	buf.WriteString("\nACTIONS: ")
	buf.WriteString(strings.Join(net.actions, ", "))
	buf.WriteString("\nFRONT NEURONS:\n")
	for _, neuron := range net.frontNeuronSet {
		buf.WriteString(neuron.String())
		buf.WriteString("\n")
	}
	buf.WriteString("\nBACK NEURONS:\n")
	for _, neuron := range net.backNeuronSet {
		buf.WriteString(neuron.String())
		buf.WriteString("\n")
	}
	buf.WriteString("\n-----\n")
	return buf.String()
}

func (net neuralnet) Save(out io.Writer) error {
	var buf bytes.Buffer
	var current [4]byte

	binary.BigEndian.PutUint16(current[:], uint16(len(net.sensors)))
	buf.Write(current[:])
	for _, sensor := range net.sensors {
		buf.Write([]byte(sensor))
		buf.Write([]byte{0})
	}
	binary.BigEndian.PutUint16(current[:], uint16(len(net.actions)))
	buf.Write(current[:])
	for _, action := range net.actions {
		buf.Write([]byte(action))
		buf.Write([]byte{0})
	}
	binary.BigEndian.PutUint16(current[:], uint16(len(net.frontNeuronSet)))
	buf.Write(current[:])
	for _, neuron := range net.frontNeuronSet {
		ch := neuron.Marshal()
		for i := 0; i < 2+4*neuron.GetSize(); i++ {
			data := <-ch
			buf.Write([]byte{data})
		}
	}
	binary.BigEndian.PutUint16(current[:], uint16(len(net.backNeuronSet)))
	buf.Write(current[:])
	for _, neuron := range net.backNeuronSet {
		ch := neuron.Marshal()
		for i := 0; i < 2+4*neuron.GetSize(); i++ {
			data := <-ch
			buf.Write([]byte{data})
		}
	}
	buf.Write([]byte{0, 0, 0, 0})

	binary.BigEndian.PutUint16(current[:], uint16(buf.Len()))
	if _, err := out.Write(current[:]); err != nil {
		return err
	}
	if _, err := out.Write(buf.Bytes()); err != nil {
		return err
	}
	return nil
}

func usort(args []string) []string {
	buf := make(map[string]bool)
	for _, arg := range args {
		buf[arg] = true
	}
	res := make([]string, len(buf))
	i := 0
	for key := range buf {
		res[i] = key
		i++
	}
	sort.Strings(res)
	return res
}
