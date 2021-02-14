<template>
  <div class='model-view border border-dark' style='height: 360px'>
    <h5 class='title bg-primary text-white'>Model View</h5>
    <div class='d-inline-flex box'>
      <button class='btn' @click="deLayer">decrease</button>
      <div class='pt-5'>
        <img class='arrow' src='../assets/arrow.png' />
      </div>
      <div class='d-inline-flex' v-for="(la, index) in arr" :key="index">
        <div class='d-flex flex-column'>
          <input type='number'
            min='0' max='500' step='1'
            v-model='neurons[index]'
            class='inputnumber p-0'>
          <img class='layer p-0' src='../assets/layer.png' />
          <p class='mb-0 p-0'>{{ la }}</p>
        </div>
        <div class='pt-5'>
          <img class='arrow' src='../assets/arrow.png' />
        </div>
      </div>
      <button class='btn' @click="inLayer">increase</button>
    </div>
    <div class='d-flex justify-content-around'>
      <div class='d-flex flex-column'>
        <button class='btn btn-light button' @click="constructModel()">construct</button>
        <div id="loss" style="width: 250px; height: 180px;"></div>
      </div>
      <div class='d-flex flex-column'>
        <div class='d-flex flex-row justify-content-center'>
          <div class='d-inline-flex'>
            <small>t:</small>
            <input type='range' id='rangeTime' min='0.0' max='1.0' value='1.0' step='0.1'
              @input="changeTime()">
            <span id="valueTime">1.0</span>
          </div>
          <p class='m-0'>| |</p>
           <button class='btn btn-light button' @click="predict()">Predict</button>
        </div>
        <div id='prediction' style='width: 365px; height: 173px'></div>
      </div>
      <div class='d-flex flex-column'>
        <div class='d-flex flex-row'>
          <div class='d-inline-flex'>
            <small>Library:</small>
            <input type='range' id='range' min='125' max='900' value='125' step='25'
              @input="change()">
            <span id="value">125</span>
          </div>
          <p class='m-0'>| |</p>
          <button class='btn btn-light button' @click='generateTS()'>Generate TS data</button>
        </div>
        <div id='timeseries' style='width: 365px; height: 173px'></div>
      </div>
    </div>
  </div>
</template>

<script>
import echarts from 'echarts';
import 'echarts-gl';
import axios from 'axios';
// import Input from './Input.vue';

export default {
  // components: { Input },
  name: 'ModelView',
  data() {
    return {
      charts: '',
      model_info: [],
      arr: [],
      neurons: [],
      loss: [],
      message: '',
      fileName: '',
      activation: '',
      xData: [],
      yData: [],
      minData: '',
      maxData: '',
      data: [],
      library: '',
      causality_lib: [],
      pred: '',
      predMin: '',
      predMax: '',
      triData: [],
      time: '',
      u: [],
      dt: [],
      dx: [],
      dy: [],
      dz: [],
      dxx: [],
      dxy: [],
      dxz: [],
      dyy: [],
      dyz: [],
      dzz: [],
      dxxx: [],
      dxxy: [],
      dxyy: [],
      dyyy: [],
      ts: [],
    };
  },
  methods: {
    deLayer() {
      this.neurons.pop();
      this.arr.pop();
    },
    inLayer() {
      const l = this.neurons.length + 1;
      const newL = 'L'.concat(l);
      this.arr.push(newL);
      this.neurons.push(10);
    },
    change() {
      this.library = document.getElementById('range').value;
      document.getElementById('value').innerHTML = this.library;
    },
    changeTime() {
      this.time = document.getElementById('rangeTime').value;
      document.getElementById('valueTime').innerHTML = this.time;
    },
    predict() {
      const payload = {
        time: this.time,
      };
      this.updateData(payload);
    },
    lossProcess() {
      const lossCopy = this.loss;
      const lossP = [];
      const dataX = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
      if (lossCopy.length !== 0) {
        const i = lossCopy.length - 1;
        for (let j = 0; j < 12; j += 1) {
          lossP.push(lossCopy[i][j]);
        }
      }
      // for (let i = 0; i < lossCopy.length; i += 1) {
      //   // if (i === 0) {
      //   //   dataX.push(0);
      //   //   lossP.push(lossCopy[0][0] * 2);
      //   // }
      //   for (let j = 0; j < 11; j += 1) {
      //     dataX.push(i + 1);
      //     lossP.push(lossCopy[i][j]);
      //   }
      // }
      return [dataX, lossP];
    },
    drawLoss(id) {
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        title: {
          text: 'loss',
        },
        tooltip: {
          trigger: 'axis',
        },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: this.lossProcess()[0],
          axisLabel: {
            interval: 0,
            showMinLabel: false,
          },
        },
        yAxis: {
          boundaryGap: [0, '50%'],
          type: 'value',
        },
        grid: {
          x: 40,
          y: 30,
          x2: 10,
          y2: 30,
        },
        series: [
          {
            name: 'loss',
            type: 'line',
            smooth: true,
            symbol: 'true',
            stack: 'a',
            data: this.lossProcess()[1],
          },
        ],
      });
    },
    draw3DScatter(id) {
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        visualMap: {
          show: false,
          min: this.predMin,
          max: this.predMax,
          inRange: {
            symbolSize: [5, 15],
            color: [
              '#313695',
              '#4575b4',
              '#74add1',
              '#abd9e9',
              '#e0f3f8',
              '#ffffbf',
              '#fee090',
              '#fdae61',
              '#f46d43',
              '#d73027',
              '#a50026',
            ],
            colorAlpha: [0.2, 1],
          },
        },
        grid3D: {
          axisLine: {
            lineStyle: { color: '#000' },
          },
          axisPointer: {
            lineStyle: { color: '#000' },
          },
          viewControl: {
            // autoRotate: true
          },
        },
        xAxis3D: {
          type: 'value',
        },
        yAxis3D: {
          type: 'value',
        },
        zAxis3D: {
          type: 'value',
        },
        series: [
          {
            type: 'scatter3D',
            data: this.triData,
          },
        ],
      });
    },
    // drawTs(id) {
    //   this.charts = echarts.init(document.getElementById(id));
    //   this.charts.setOption({
    //     // title: {
    //     //   text: 'Error',
    //     // },
    //     tooltip: {
    //       trigger: 'axis',
    //     },
    //     legend: {
    //       top: '3%',
    //       data: ['u', 'dt', 'dx', 'dy', 'dz', 'dxx', 'dxy', 'dxz', 'dyy', 'dyz', 'dzz'],
    //     },
    //     grid: {
    //       left: '2%',
    //       right: '4%',
    //       bottom: '3%',
    //       top: '35%',
    //       containLabel: true,
    //     },
    //     // toolbox: {
    //     //   feature: {
    //     //     saveAsImage: {},
    //     //   },
    //     // },
    //     xAxis: {
    //       type: 'category',
    //       boundaryGap: false,
    //       data: this.ts,
    //     },
    //     yAxis: {
    //       type: 'value',
    //     },
    //     series: [
    //       {
    //         name: 'u',
    //         type: 'line',
    //         data: this.u,
    //       },
    //       {
    //         name: 'dt',
    //         type: 'line',
    //         data: this.dt,
    //       },
    //       {
    //         name: 'dx',
    //         type: 'line',
    //         data: this.dx,
    //       },
    //       {
    //         name: 'dy',
    //         type: 'line',
    //         data: this.dy,
    //       },
    //       {
    //         name: 'dz',
    //         type: 'line',
    //         data: this.dz,
    //       },
    //       {
    //         name: 'dxx',
    //         type: 'line',
    //         data: this.dxx,
    //       },
    //       {
    //         name: 'dxy',
    //         type: 'line',
    //         data: this.dxy,
    //       },
    //       {
    //         name: 'dxz',
    //         type: 'line',
    //         data: this.dxz,
    //       },
    //       {
    //         name: 'dyy',
    //         type: 'line',
    //         data: this.dyy,
    //       },
    //       {
    //         name: 'dyz',
    //         type: 'line',
    //         data: this.dyz,
    //       },
    //       {
    //         name: 'dzz',
    //         type: 'line',
    //         data: this.dzz,
    //       },
    //     ],
    //   });
    // },
    drawTs(id) {
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        // title: {
        //   text: 'Error',
        // },
        tooltip: {
          trigger: 'axis',
        },
        legend: {
          top: '3%',
          data: ['u', 'dt', 'dx', 'dy', 'dxx', 'dxy', 'dyy', 'dxxx', 'dxxy', 'dxyy', 'dyyy'],
        },
        grid: {
          left: '2%',
          right: '4%',
          bottom: '3%',
          top: '35%',
          containLabel: true,
        },
        // toolbox: {
        //   feature: {
        //     saveAsImage: {},
        //   },
        // },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: this.ts,
        },
        yAxis: {
          type: 'value',
        },
        series: [
          {
            name: 'u',
            type: 'line',
            data: this.u,
          },
          {
            name: 'dt',
            type: 'line',
            data: this.dt,
          },
          {
            name: 'dx',
            type: 'line',
            data: this.dx,
          },
          {
            name: 'dy',
            type: 'line',
            data: this.dy,
          },
          {
            name: 'dz',
            type: 'line',
            data: this.dz,
          },
          {
            name: 'dxx',
            type: 'line',
            data: this.dxx,
          },
          {
            name: 'dxy',
            type: 'line',
            data: this.dxy,
          },
          {
            name: 'dxz',
            type: 'line',
            data: this.dxz,
          },
          {
            name: 'dyy',
            type: 'line',
            data: this.dyy,
          },
          {
            name: 'dyz',
            type: 'line',
            data: this.dyz,
          },
          {
            name: 'dzz',
            type: 'line',
            data: this.dzz,
          },
        ],
      });
    },
    getData() {
      const path = 'http://localhost:5000/prediction';
      axios.get(path)
        .then((res) => {
          this.pred = res.data.prediction;
          this.message = res.data.status;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    updateData(payload) {
      const path = 'http://localhost:5000/prediction';
      axios.post(path, payload)
        .then(() => {
          this.getData();
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getData();
        });
    },
    getModelInfo() {
      const path = 'http://localhost:5000/model';
      axios.get(path)
        .then((res) => {
          this.model_info = res.data.model_info;
          this.message = res.data.status;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    updateModelInfo(payload) {
      const path = 'http://localhost:5000/model';
      axios.post(path, payload)
        .then(() => {
          this.getModelInfo();
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getModelInfo();
        });
    },
    constructModel() {
      const payload = {
        fileName: this.fileName,
        activation: this.activation,
        layers: this.arr.length,
        neurons: this.neurons,
      };
      this.updateModelInfo(payload);
    },
    getCausalityInfo() {
      const path = 'http://localhost:5000/causality_lib';
      axios.get(path)
        .then((res) => {
          this.causality_lib = res.data.causality_lib;
          this.message = res.data.status;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    updateCausalityInfo(payload) {
      const path = 'http://localhost:5000/causality_lib';
      axios.post(path, payload)
        .then(() => {
          this.getCausalityInfo();
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getCausalityInfo();
        });
    },
    generateTS() {
      const payload = {
        library: this.library,
      };
      this.updateCausalityInfo(payload);
    },
  },
  created() {
    const init = setInterval(() => {
      this.getModelInfo();
      if (this.message === 'success') {
        this.fileName = this.model_info.fileName;
        this.activation = this.model_info.activation;
        this.neurons = this.model_info.neurons;
        this.loss = this.model_info.loss;
        this.arr = this.model_info.arr;
        this.drawLoss('loss');
        clearInterval(init);
      }
    }, 1000);
  },
  mounted() {
    this.drawLoss('loss');
    this.draw3DScatter('prediction');
    this.drawTs('timeseries');
  },
  watch: {
    model_info: {
      handler(newValue, oldValue) {
        if (newValue !== oldValue) {
          this.neurons = newValue.neurons;
          this.arr = newValue.arr;
          this.loss = newValue.loss;
          this.drawLoss('loss');
        }
      },
      deep: true,
    },
    pred: {
      handler(newValue, oldValue) {
        if (newValue !== oldValue) {
          this.predMin = newValue.predMin;
          this.predMax = newValue.predMax;
          this.triData = newValue.triData;
          this.draw3DScatter('prediction');
        }
      },
      deep: true,
    },
    causality_lib: {
      handler(newValue, oldValue) {
        if (newValue !== oldValue) {
          // this.u = newValue.u;
          // this.dt = newValue.dt;
          // this.dx = newValue.dx;
          // this.dy = newValue.dy;
          // this.dz = newValue.dz;
          // this.dxx = newValue.dxx;
          // this.dxy = newValue.dxy;
          // this.dxz = newValue.dxz;
          // this.dyy = newValue.dyy;
          // this.dyz = newValue.dyz;
          // this.dzz = newValue.dzz;
          // this.ts = newValue.ts;
          this.u = newValue.u;
          this.dt = newValue.dt;
          this.dx = newValue.dx;
          this.dy = newValue.dy;
          this.dxx = newValue.dxx;
          this.dxy = newValue.dxy;
          this.dyy = newValue.dyy;
          this.dxxx = newValue.dxx;
          this.dxxy = newValue.dxxy;
          this.dxyy = newValue.dxyy;
          this.dyyy = newValue.dyyy;
          this.ts = newValue.ts;
          this.drawTs('timeseries');
        }
      },
      deep: true,
    },
  },
};
</script>

<style>
.arrow {
  width: 50px;
  height: 10px;
}
.layer {
  width: 30px;
  height: 80px;
}
.inputnumber {
  width: 30px;
  height: 20px;
  font-size: 13px;
  text-align: center;
}
input::-webkit-outer-spin-button,
input::-webkit-inner-spin-button {
    -webkit-appearance: none !important;
    margin: 0;
}
.box{
  width: 1100px;
  height: 130px;
  display: flex;
  display: -webkit-box;
  white-space: nowrap;
  overflow: hidden;
  overflow-x: initial;
}
.button {
  height: 17px;
  line-height: 17px;
  padding: 0px;
  text-align: center;
}
#candidate {
  height: 17px;
  width: 60px;
  line-height: 17px;
  padding: 0px;
  text-align: center;
  font-size: 12px;
}
</style>
