package neuron

import (
	"bytes"
	"encoding/base32"
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"strings"
)

// Neuron represents a neuron
type Neuron interface {
	Compute(...float64) int
	Equals(Neuron) bool
	GetSize() int
	GetGene(int) int
	Marshal() <-chan byte
	Child(int) Neuron
	String() string
}

type neuron []int

// NewNeuron create a new neuron
func NewNeuron(data interface{}) (Neuron, error) {

	switch value := data.(type) {
	case []int:
		neu := make(neuron, len(value))
		copy(neu, value)
		return neu, nil

	case Neuron:
		return value, nil

	case []byte:
		return neuronFromBytes(value)

	case io.Reader:
		return readFile(value)

	case *bytes.Buffer:
		return neuronFromBytes(value.Bytes())

	case int:
		neu := make(neuron, value)
		for i := 0; i < value; i++ {
			neu[i] = int(rand.Int31n(2000)) - 1000
		}
		return neu, nil

	case string:
		decoder := base32.HexEncoding.WithPadding(base32.NoPadding)
		data, err := decoder.DecodeString(strings.TrimSpace(value))
		if err == nil {
			return neuronFromBytes(data)
		}
		return nil, err

	default:
		return nil, fmt.Errorf("unexpected type %T", value)
	}
}

func (neu neuron) GetSize() int {
	return len(neu)
}

func (neu neuron) GetGene(index int) int {
	return neu[index]
}

func (neu neuron) Equals(other Neuron) bool {
	if neu.GetSize() != other.GetSize() {
		return false
	}
	for i, value := range neu {
		if value != other.GetGene(i) {
			return false
		}
	}
	return true
}

func (neu neuron) Compute(data ...float64) int {
	if len(data) != neu.GetSize() {
		panic(fmt.Sprintf("expected %v parameters, got %v", neu.GetSize(), len(data)))
	}

	sum := 0.0

	for index, value := range data {
		sum += value * float64(neu.GetGene(index))
	}

	if sum > 0 {
		return int(sum)
	}
	return 0
}

func (neu neuron) Child(dev int) Neuron {
	child := make(neuron, neu.GetSize())
	for i, value := range neu {
		child[i] = value + int(rand.Int31n(int32(dev))) - (dev / 2)
	}
	return child
}

func (neu neuron) Marshal() <-chan byte {
	ch := make(chan byte)

	go func() {
		size := neu.GetSize()
		var buf [4]byte
		binary.BigEndian.PutUint16(buf[:], uint16(size))
		ch <- buf[0]
		ch <- buf[1]

		for _, gene := range neu {
			var cur uint32
			if gene < 0 {
				cur = uint32(int64(0x100000000) + int64(gene))
			} else {
				cur = uint32(gene)
			}
			binary.BigEndian.PutUint32(buf[:], cur)
			for i := 0; i < 4; i++ {
				ch <- buf[i]
			}
		}
	}()

	return ch
}

func (neu neuron) String() string {
	size := 2 + 4*neu.GetSize()
	buf := make([]byte, size)
	ch := neu.Marshal()
	for i := 0; i < size; i++ {
		buf[i] = <-ch
	}
	encoder := base32.HexEncoding.WithPadding(base32.NoPadding)
	return encoder.EncodeToString(buf)
}

func readFile(input io.Reader) (Neuron, error) {
	var buf [2]byte
	if _, err := input.Read(buf[:]); err != nil {
		return nil, err
	}
	size := 4 * int(binary.BigEndian.Uint16(buf[:]))
	data := make([]byte, 2+size)
	copy(data, buf[:])
	if _, err := input.Read(data[2:]); err != nil {
		return nil, err
	}
	return neuronFromBytes(data)
}

func neuronFromBytes(input []byte) (Neuron, error) {
	res := make(chan int)
	ech := make(chan error)

	size := int(binary.BigEndian.Uint16(input))
	go processBytes(input[2:], size, res, ech)
	neu := make(neuron, size)

	for i := 0; i < size; {
		select {
		case err := <-ech:
			return nil, err
		case value := <-res:
			neu[i] = value
			i++
		default:
		}
	}

	return neu, nil
}

func processBytes(body []byte, size int, res chan<- int, ech chan<- error) {
	defer func() {
		if err, ok := recover().(error); ok {
			ech <- err
		}
	}()

	for i := 0; i < size; i++ {
		value := binary.BigEndian.Uint32(body[i*4:])
		if value >= 0x80000000 {
			res <- int(int64(value) - int64(0x100000000))
		} else {
			res <- int(value)
		}
	}
}
