package neuron

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"sort"
	"strings"
)

// Layer represents a layer of neurons
type Layer []Neuron

// NeuralNet represents a neural net
type NeuralNet interface {
	GetActions() []string
	GetChild(int) NeuralNet
	GetSensors() []string
	GetNeurons(int) []Neuron
	Compute(map[string]float64) (map[string]bool, error)
	Save(io.Writer) error
	String() string
}

type neuralnet struct {
	actions []string
	neurons []Layer
	sensors []string
}

// NewNeuralNet instantiate a new neural net
func NewNeuralNet(sensors, actions []string, neurons []Layer) (NeuralNet, error) {
	if len(neurons) == 0 {
		return nil, fmt.Errorf("no neuron supplied")
	}

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

	count := sensorsCount
	var last []Neuron

	for i, current := range neurons {
		for j, neuron := range current {
			if neuron.GetSize() != count {
				return nil, fmt.Errorf("group %v, neuron %v: expected size %v, got %v", i, j, count, neuron.GetSize())
			}
		}
		count = len(current)
		last = current
	}
	if len(last) != actionsCount {
		return nil, fmt.Errorf("expected one last neuron [%v] for each action [%v]", len(last), actionsCount)
	}

	return &neuralnet{sortedActions, neurons, sortedSensors}, nil
}

// LoadNet load a new neural net from an I/O reader
func LoadNet(input io.Reader) (NeuralNet, error) {
	var buf [4]byte

	// Discard long size
	if _, err := input.Read(buf[:]); err != nil {
		return nil, err
	}

	var err error
	var sensors []string
	var actions []string
	var neurons []Layer

	if sensors, err = loadStrings(input); err != nil {
		return nil, err
	}
	if actions, err = loadStrings(input); err != nil {
		return nil, err
	}

	if _, err := input.Read(buf[:]); err != nil {
		return nil, err
	}
	size := int(binary.BigEndian.Uint16(buf[:]))
	neurons = make([]Layer, size)

	for i := 0; i < size; i++ {
		if current, err := loadNeurons(input); err == nil {
			neurons[i] = current
		} else {
			return nil, err
		}
	}

	// Put everything together
	return NewNeuralNet(sensors, actions, neurons)
}

func (net neuralnet) GetChild(dev int) NeuralNet {
	neurons := make([]Layer, len(net.neurons))
	for i, layer := range net.neurons {
		current := make(Layer, len(layer))
		for j, neuron := range layer {
			current[j] = neuron.Child(dev)
		}
		neurons[i] = current
	}
	return &neuralnet{net.actions, neurons, net.sensors}
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

func (net neuralnet) GetNeurons(index int) []Neuron {
	if index >= len(net.neurons) {
		return nil
	}
	neurons := make([]Neuron, len(net.neurons[index]))
	copy(neurons, net.neurons[index])
	return neurons
}

func (net neuralnet) Compute(incoming map[string]float64) (map[string]bool, error) {
	if err := net.checkInput(incoming); err != nil {
		return nil, err
	}

	partial := make([]float64, len(incoming))
	for i, sensor := range net.sensors {
		partial[i] = incoming[sensor]
	}

	for _, neurons := range net.neurons {
		nextStep := make([]float64, len(neurons))
		for i, neuron := range neurons {
			nextStep[i] = float64(neuron.Compute(partial...))
		}
		partial = nextStep
	}

	res := make(map[string]bool)
	for i, action := range net.actions {
		res[action] = partial[i] > 0
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
	buf.WriteString("\nNEURONS:\n")
	for _, neurons := range net.neurons {
		for _, neuron := range neurons {
			buf.WriteString(neuron.String())
			buf.WriteByte(0x0a)
		}
		buf.WriteByte(0x0a)
	}
	buf.WriteString("-----\n")
	return buf.String()
}

func (net neuralnet) Save(out io.Writer) error {
	var buf bytes.Buffer
	var current [4]byte

	// Serialise sensors
	binary.BigEndian.PutUint16(current[:], uint16(len(net.sensors)))
	buf.Write(current[:])
	for _, sensor := range net.sensors {
		buf.Write([]byte(sensor))
		buf.WriteByte(0x00)
	}

	// Serialise actions
	binary.BigEndian.PutUint16(current[:], uint16(len(net.actions)))
	buf.Write(current[:])
	for _, action := range net.actions {
		buf.Write([]byte(action))
		buf.WriteByte(0x00)
	}

	// Serialise neurons
	binary.BigEndian.PutUint16(current[:], uint16(len(net.neurons)))
	buf.Write(current[:])
	for _, neurons := range net.neurons {
		binary.BigEndian.PutUint16(current[:], uint16(len(neurons)))
		buf.Write(current[:])
		for _, neuron := range neurons {
			ch := neuron.Marshal()
			value, ok := <-ch
			for ok {
				buf.WriteByte(value)
				value, ok = <-ch
			}
		}
	}

	// Add tail
	buf.Write([]byte{0, 0, 0, 0})

	// Add header
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

func loadNeurons(input io.Reader) ([]Neuron, error) {
	var buf [4]byte
	if _, err := input.Read(buf[:]); err != nil {
		return nil, err
	}
	size := int(binary.BigEndian.Uint16(buf[:]))
	res := make([]Neuron, size)

	for i := 0; i < size; i++ {
		var err error
		res[i], err = NewNeuron(input)
		if err != nil {
			return nil, err
		}
	}
	return res, nil
}

func loadStrings(input io.Reader) ([]string, error) {
	var buf [4]byte
	if _, err := input.Read(buf[:]); err != nil {
		return nil, err
	}
	size := int(binary.BigEndian.Uint16(buf[:]))
	set := make([]string, size)

	for i := 0; i < size; i++ {
		current := []byte{0xff}
		var str strings.Builder
		for current[0] != 0 {
			if _, err := input.Read(current[:]); err != nil {
				return nil, err
			}
			if current[0] != 0x00 {
				str.WriteByte(current[0])
			}
		}
		set[i] = str.String()
	}
	return set, nil
}
