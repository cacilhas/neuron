[base32hex]: https://tools.ietf.org/html/rfc4648#page-10
[bsdv3]: https://opensource.org/licenses/BSD-3-Clause
[golang]: https://golang.org/

# Neuron – Neural Network Support

Build a neural network over [Go][golang].

## Installation

```sh
gvt fetch github.com/cacilhas/neuron
```

## Use

```go
sensors := []string{
	"distance",
	"height",
}

actions := []string{"jump"}

buildNeuron := func(length int) neuron.Neuron {
	neu, err := neuron.NewNeuron(length)
	if err != nil {
		panic(err)
	}
	return neu
}

front := neuron.Layer{buildNeuron(2), buildNeuron(2), buildNeuron(2)}
middle := neuron.Layer{buildNeuron(3), buildNeuron(3)}
back := neuron.Layer{buildNeuron(2)}

net := neuron.NewNeuralNet(sensors, actions, []neuron.Layer{front, middle, back})
```

Using the network:

```go
params := []float64{-12, 246.125}
res, _ := net.Compute(params)

if res["jump"] {
	// Jump
}
```

### Saving and retrieving

Save to file:

```go
fp, err := os.Create("net.dna")
if err != nil {
	panic(err)
}
defer fp.Close()

err = net.Save(fp)
if err != nil {
	panic(err)
}
```

Load from file:

```go
fp, err := os.Open("net.dna")
if err != nil {
	panic(err)
}
defer fp.Close()

var net neuron.NeuralNet
net, err := neuron.LoadNet(fp)
if err != nil {
	panic(err)
}
```

### API

`Neuron` (`neuron` is the instance):

- `NewNeuron([]byte) (Neuron, error)`
- `NewNeuron(*bytes.Buffer) (Neuron, error)`
  - Build a neuron from its binary representation.
- `NewNeuron([]int) (Neuron, error)`
  - Build a new neuron from the `int` array. Each integer represents a gene.
- `NewNeuron(int) (Neuron, error)`
  - Build a new random neuron, `int` is the amount of genes.
- `NewNeuron(io.Reader) (Neuron, error)`
  - Load a neuron from a stream.
- `NewNeuron(Neuron) (Neuron, error)`
  - Clone the neuron supplied.
- `NewNeuron(string) (Neuron, error)`
  - Deserialise a neuron.
- `neuron.Compute(...float64) int`
  - Compute the output from a list of parameters. There must be supplied as many parameters as genes.
- `neuron.Equals(Neuron) bool`
  - Check whether two neurons have the save genetic pool.
- `neuron.GetSize() int`
  - Return the neuron size (amount of genes).
- `neuron.GetGene(int) int`
  - Return the value of the gene in the index `int`.
- `neuron.Marshal() <-chan byte`
  - Return a channel that supplies the neuron binary representation byte by byte.
- `neuron.Child(int) Neuron`
  - Return a new random child neuron, with the deviation `int`.
- `neuron.String() string`
  - Return the neuron binary representation encoded on [base32hex][base32hex].

`NeuralNet` (`net` is the instance):

- `NewNeuralNet(sensors, actions []string, neurons []Layer) (NeuralNet, error)`
  - Create a new neural network given the parameters.
- `LoadNet(io.Reader) (NeuralNet, error)`
  - Load a neural network from a stream.
- `net.GetActions() []string`
  - Return the neural network’s actions.
- `net.Compute(map[string]float64) (map[string]bool, error)`
  - Compute the processing. The `map[string]float64` parameter must supply one key for each network’s sensor, and the `map[string]bool` brings if each action must be performed.
- `net.Neurons(index) []Neuron`
  - Return the neural network’s `int` layer of neurons (`nil` if `int` is too big).
- `net.GetSensors() []string`
  - Return the neural network’s sensors.
- `net.Save(io.Writer) error`
  - Save the neural network into a stream.
- `net.String() string`
  - Return the neural network serialisation.

`neuron.Layer` is a layer of neurons:

```go
type Layer []Neuron
```

## License

Copyright 2019 Rodrigo Cacilhas <batalema@cacilhas.info>

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

[The 3-Clause BSD License][bsdv3]
