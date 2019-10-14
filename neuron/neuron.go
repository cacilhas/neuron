package neuron

import (
	"bytes"
	"encoding/hex"
	"fmt"
)

// Neuron represents a neuron
type Neuron interface {
	GetSize() int
	GetGen(int) int
	Compute(...int) int
	Marshal() *bytes.Buffer
	String() string
}

type neuron []int

// NewNeuron create a new neuron
func NewNeuron(data interface{}) (Neuron, error) {

	switch v := data.(type) {
	case []int:
		var n neuron
		copy(n, v)
		return n, nil

	case neuron:
		return v, nil

	case []byte:
		return neuronFromBytes(v)

	case *bytes.Buffer:
		return neuronFromBytes(v.Bytes())

	case string:
		src, err := hex.DecodeString(v)
		if err == nil {
			return neuronFromBytes(src)
		}
		return nil, err

	default:
		return nil, fmt.Errorf("unexpected type %T", v)
	}
}

func neuronFromBytes(data []byte) (Neuron, error) {
	if len(data) < 4 {
		return nil, fmt.Errorf("invalid input")
	}

	size := 0
	for i := 0; i < 4; i++ {
		size += int(data[i]) << uint(8*i)
	}

	if len(data) < (size+1)*4 {
		return nil, fmt.Errorf("invalid input")
	}

	buf := make([]int, size)
	for i := 0; i < size; i++ {
		for j := 0; j < 4; j++ {
			index := (i+1)*4 + j
			buf[i] += int(data[index]) << uint(8*j)
		}
	}

	return neuron(buf), nil
}

func (n neuron) GetSize() int {
	return len(n)
}

func (n neuron) GetGen(index int) int {
	return n[index]
}

func (n neuron) Compute(data ...int) int {
	if len(data) != len(n) {
		panic(fmt.Sprintf("wrong number or parameters, expect %v, got %v", len(n), len(data)))
	}

	res := 0
	for index, value := range n {
		res += data[index] * value
	}

	if res > 0 {
		return res
	}
	return 0
}

func (n neuron) Marshal() *bytes.Buffer {
	var buf bytes.Buffer
	size := len(n)
	for i := 0; i < 4; i++ {
		buf.WriteByte(byte((size >> uint(8*i)) & 0xff))
	}
	for _, value := range n {
		for i := 0; i < 4; i++ {
			buf.WriteByte(byte((value >> uint(8*i)) & 0xff))
		}
	}
	return bytes.NewBuffer(buf.Bytes())
}

func (n neuron) String() string {
	return hex.EncodeToString(n.Marshal().Bytes())
}
