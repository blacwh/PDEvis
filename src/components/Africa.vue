<template>
  <div class='column'>
    <div class='row mb-5'>
        <div id="africa1" style="width: 500px; height: 500px"></div>
        <div id="africa2" style="width: 500px; height: 500px"></div>
    </div>
    <div class='d-flex justify-content-around'>
      <div class='d-inline-flex'>
        <small>time:</small>
        <input type='range' id='rangeT' min='0' max='371' value='0' step='1'
          @input="change()"
          style="width: 500px;">
        <span id="valueT">0</span>
      </div>
      <button class='btn btn-light button' @click='refresh()'>Refresh</button>
    </div>
  </div>
</template>

<script>
import echarts from 'echarts';
import axios from 'axios';
import africa from '../assets/custom.geo.json';

export default {
  name: 'Africa',
  data() {
    return {
      charts: '',
      dataset: africa,
      africa: '',
      name: '',
      real: [],
      pred: [],
      real_min: '',
      real_max: '',
      pred_min: '',
      pred_max: '',
      valueT: '',
    };
  },
  methods: {
    refresh() {
      const payload = {
        valueT: this.valueT,
      };
      this.updateCoronaData(payload);
    },
    change() {
      this.valueT = document.getElementById('rangeT').value;
      document.getElementById('valueT').innerHTML = this.valueT;
    },
    drawMap(id) {
      echarts.registerMap('africa', this.dataset);
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        title: {
          text: 'Africa Corona Infection (2020)',
          subtext: 'Data from CSSE at Johns Hopkins University',
          sublink: 'https://github.com/CSSEGISandData/COVID-19',
          left: 'right',
        },
        // tooltip: {
        //   trigger: 'item',
        //   showDelay: 0,
        //   transitionDuration: 0.2,
        //   formatter: function rename(params) {
        //     this.name = params.value;
        //     let value = `${params.value} `.split('.');
        //     value = value[0].replace(/(\d{1,3})(?=(?:\d{3})+(?!\d))/g, '$1,');
        //     return `${params.seriesName}<br/>${params.name}: ${value}`;
        //   },
        // },
        visualMap: {
          left: 'right',
          min: this.real_min,
          max: this.real_max,
          //   min: 0,
          //   max: 1430648,
          inRange: {
            color: ['#313695',
              '#4575b4',
              '#74add1',
              '#abd9e9',
              '#e0f3f8',
              '#ffffbf',
              '#fee090',
              '#fdae61',
              '#f46d43',
              '#d73027',
              '#a50026'],
          },
          text: ['High', 'Low'],
          calculable: true,
        },
        // toolbox: {
        //   show: true,
        //   //  orient: 'vertical',
        //   left: 'left',
        //   top: 'top',
        //   feature: {
        //     dataView: { readOnly: false },
        //     restore: {},
        //     saveAsImage: {},
        //   },
        // },
        series: [
          {
            name: 'Africa infection',
            type: 'map',
            roam: true,
            map: 'africa',
            emphasis: {
              label: {
                show: true,
              },
            },
            // 文本位置修正
            textFixed: {
              Alaska: [20, -20],
            },
            data: this.real,
            // data: [{ name: 'Egypt', value: 380000 }],
            //   { name: 'Alabama', value: 4822023 },
          },
        ],
      });
    },
    drawMap2(id) {
      echarts.registerMap('africa', this.dataset);
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        title: {
          text: 'Africa Corona Infection Estimation',
          //   subtext: 'Data from CSSE at Johns Hopkins University',
          left: 'right',
        },
        // tooltip: {
        //   trigger: 'item',
        //   showDelay: 0,
        //   transitionDuration: 0.2,
        //   formatter: function rename(params) {
        //     this.name = params.value;
        //     let value = `${params.value} `.split('.');
        //     value = value[0].replace(/(\d{1,3})(?=(?:\d{3})+(?!\d))/g, '$1,');
        //     return `${params.seriesName}<br/>${params.name}: ${value}`;
        //   },
        // },
        visualMap: {
          left: 'right',
          min: this.pred_min,
          max: this.pred_max,
          //   min: 0,
          //   max: 1430648,
          inRange: {
            color: ['#313695',
              '#4575b4',
              '#74add1',
              '#abd9e9',
              '#e0f3f8',
              '#ffffbf',
              '#fee090',
              '#fdae61',
              '#f46d43',
              '#d73027',
              '#a50026'],
          },
          text: ['High', 'Low'],
          calculable: true,
        },
        series: [
          {
            name: 'Africa infection',
            type: 'map',
            roam: true,
            map: 'africa',
            emphasis: {
              label: {
                show: true,
              },
            },
            // 文本位置修正
            textFixed: {
              Alaska: [20, -20],
            },
            data: this.pred,
          },
        ],
      });
    },
    getCoronaData() {
      const path = 'http://localhost:5000/africa';
      axios.get(path)
        .then((res) => {
          this.africa = res.data.africa;
          this.message = res.data.status;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    updateCoronaData(payload) {
      const path = 'http://localhost:5000/africa';
      axios
        .post(path, payload)
        .then(() => {
          this.getCoronaData();
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getCoronaData();
        });
    },
  },
  mounted() {
    this.drawMap('africa1');
    this.drawMap2('africa2');
  },
  watch: {
    africa: {
      handler(newValue, oldValue) {
        if (newValue !== oldValue) {
          this.real = newValue.real;
          this.pred = newValue.pred;
          this.real_min = newValue.real_min;
          this.real_max = newValue.real_max;
          this.pred_min = newValue.pred_min;
          this.pred_max = newValue.pred_max;
          this.drawMap('africa1');
          this.drawMap2('africa2');
        }
      },
      deep: true,
    },
  },
};
</script>
