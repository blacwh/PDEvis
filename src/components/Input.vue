<template>
  <div class='input-panel border border-dark'>
    <h5 class='title bg-primary text-white'>Input panel</h5>
    <div class='d-flex justify-content-around mb-1'>
      <input
        type='file'
        name='file'
        id='files'
        accept='.csv'
        v-on:change='fileLoad()'>
      <div class='inline'>
        <input type="checkbox" id='checkbox1'>Neural network
      </div>
      <div class='control font-weight-light small' style='width:100px'>
        <label for="layers" class='m-0 d-flex flex-row'>Layers:</label>
        <div class='select'>
          <select class="form-control" id="layers">
            <option value='1'>1</option>
            <option value='2'>2</option>
            <option value='3'>3</option>
            <option value='4'>4</option>
            <option value='5'>5</option>
            <option value='6'>6</option>
            <option value='7'>7</option>
            <option value='8'>8</option>
            <option value='9'>9</option>
            <option value='10'>10</option>
          </select>
        </div>
      </div>
      <div class='control font-weight-light small' style='width:100px'>
        <label for="neurons" class='m-0 d-flex flex-row'>Neurons:</label>
        <div class='select'>
          <select class="form-control" id="neurons">
            <option value='10'>10</option>
            <option value='20'>20</option>
            <option value='30'>30</option>
            <option value='40'>40</option>
            <option value='50'>50</option>
          </select>
        </div>
      </div>
      <div class='control font-weight-light small' style='width:100px'>
        <label for="activation" class='m-0 d-flex flex-row'>Activation:</label>
        <div class='select'>
          <select class="form-control" id="activation">
            <option value='sigmoid'>sigmoid</option>
            <option value='sin'>sin</option>
            <option value='tanh'>tanh</option>
          </select>
        </div>
      </div>
      <button class='btn btn-dark' id='runBtn' @click="runTraining()">
        <icon name="play"></icon>
      </button>
    </div>
  </div>
</template>

<script>
import axios from 'axios';

export default {
  name: 'InputPanel',
  data() {
    return {
      model_info: [],
      fileName: '',
      layers: [],
      neurons: [],
      activation: [],
      neuralNetwork: false,
    };
  },
  methods: {
    fileLoad() {
      this.fileName = document.getElementById('files').files[0].name;
    },
    getModel() {
      const path = 'http://localhost:5000/model';
      axios.get(path)
        .then((res) => {
          this.model_info = res.data.model_info;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    initModel(payload) {
      const path = 'http://localhost:5000/model';
      axios.post(path, payload)
        .then(() => {
          this.getModel();
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getModel();
        });
    },
    runTraining() {
      this.neuralNetwork = document.getElementById('checkbox1').checked;
      if (this.neuralNetwork === true && this.fileName !== '') {
        const myLayers = document.getElementById('layers');
        const idx1 = myLayers.selectedIndex;
        const l = Number(myLayers.options[idx1].value);
        this.layers = l;
        const myNeurons = document.getElementById('neurons');
        const idx2 = myNeurons.selectedIndex;
        const n = new Array(l);
        for (let i = 0; i < l; i += 1) {
          n[i] = Number(myNeurons.options[idx2].value);
        }
        this.neurons = n;
        const myActivation = document.getElementById('activation');
        const idx3 = myActivation.selectedIndex;
        this.activation = myActivation.options[idx3].value;
        const payload = {
          fileName: this.fileName,
          layers: this.layers,
          neurons: this.neurons,
          activation: this.activation,
        };
        this.initModel(payload);
      }
    },
  },
};
</script>

<style>
#layers, #neurons, #activation {
  font-size: 12px;
  border-bottom: 1px solid black;
  border-radius: 0%;
  border-top: 0;
  border-left: 0;
  border-right: 0;
}
#runBtn {
  border-radius: 50%;
  width: 37px;
  height: 37px;
}
</style>
