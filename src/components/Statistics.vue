<template>
  <div class='statistics-view border border-dark' style='height: 615px'>
    <h5 class='title bg-primary text-white'>Statistics View</h5>
    <div class='column'>
      <button class='btn btn-light button' @click='refresh()'>
        Plot
      </button>
      <div class='d-flex justify-content-around'>
        <div id="heatmap" style="width: 400px; height: 400px"></div>
        <div id="heatmap2" style="width: 400px; height: 400px"></div>
      </div>
      <!-- <div id='causality' style='height: 300px'></div> -->
      <!-- <div id='transfer_linear' style='width: 250px; height: 100px;'></div>
      <div id='transfer_nonlinear' style='width: 250px; height: 250px;'></div> -->
      <div class='d-flex justify-content-around'>
        <p class='mb-0' style='font-size: 24px'>PDE derivation:</p>
        <div class='mt-2'>
          <input
            type='radio'
            name='regression-type'
            value='Ridge'
            @change='changeRegressionRidge'
          />Ridge
          <input
            type='radio'
            name='regression-type'
            value='Lasso'
            @change='changeRegressionLasso'
          />Lasso
          <input
            type='radio'
            name='regression-type'
            value='Elasticnet'
            @change='changeRegressionElasticnet'
          />Elasticnet
        </div>
      </div>
      <p class='mb-0'>
        du_dt =<!--
          +(-1.099974)*u_x+
          (-2.139241)*u_y+(-0.986073)*u_z+(1.051358)*u_xx+(0.986085)*u_yy+(0.981638)*u_zz
        -->
        <span v-for='(value, index) in coef' :key='index'
          >{{
            '+' + '(' + coef[index].toFixed(2) + ')*' + candidates_left[index]
          }}
        </span>
      </p>
    </div>
  </div>
</template>

<script>
import echarts from 'echarts';
import axios from 'axios';

export default {
  name: 'StatisticsView',
  data() {
    return {
      charts: '',
      stat: [],
      arr1: [],
      arr2: [],
      transfer_linear: [],
      transfer_nonlinear: [],
      regressionType: '',
      candidates_left: [],
      coef: [],
      heatmap: '',
      xData: [],
      yData: [],
      dataMax: '',
      dataMin: '',
      dataHeat: [],
      dataMax2: '',
      dataMin2: '',
      dataHeat2: [],
    };
  },
  methods: {
    refresh() {
      this.getHeatmapData();
    },
    drawHeatmap(id) {
      /*
      const dataHeat = [
        [0, 0, 10.3030807],
        [0, 1, 8.93165146],
        [0, 2, 7.56022224],
        [0, 3, 6.18879302],
        [0, 4, 4.81736381],
        [1, 0, 14.085559],
        [1, 1, 11.8850568],
        [1, 2, 9.68455459],
        [1, 3, 7.48405236],
        [1, 4, 5.28355014],
        [2, 0, 17.8680374],
        [2, 1, 14.8384622],
        [2, 2, 11.8088869],
        [2, 3, 8.7793117],
        [2, 4, 5.74973647],
        [3, 0, 21.6505158],
        [3, 1, 17.7918675],
        [3, 2, 13.9332193],
        [3, 3, 10.074571],
        [3, 4, 6.21592281],
        [4, 0, 25.4329941],
        [4, 1, 20.7452729],
        [4, 2, 16.0575516],
        [4, 3, 11.3698304],
        [4, 4, 6.68210914],
      ];
      const dataMax = 25.4329941;
      const dataMin = 4.81736381;
      */
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        title: {
          text: 'NN model accuracy',
        },
        tooltip: {},
        xAxis: {
          type: 'category',
          name: 'neurons',
          nameLocation: 'center',
          nameTextStyle: {
            verticalAlign: 'top',
            padding: [4, 4, 4, 4],
          },
          axisLabel: {
            interval: 9,
            showMinLabel: false,
          },
          data: this.xData,
        },
        yAxis: {
          type: 'category',
          data: this.yData,
          name: 'layers',
          nameLocation: 'center',
          nameTextStyle: {
            verticalAlign: 'bottom',
            padding: [4, 4, 4, 4],
          },
          axisLabel: {
            interval: 9,
            showMinLabel: false,
          },
        },
        visualMap: {
          min: this.dataMin,
          max: this.dataMax,
          calculable: true,
          realtime: false,
          inRange: {
            color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'],
          },
        },
        series: [{
          type: 'heatmap',
          data: this.dataHeat,
          emphasis: {
            itemStyle: {
              borderColor: '#333',
              borderWidth: 1,
            },
          },
          progressive: 1000,
          animation: false,
        }],
      });
    },
    drawHeatmap2(id) {
      /*
      const dataHeat = [
        [0, 0, 10.3030807],
        [0, 1, 8.93165146],
        [0, 2, 7.56022224],
        [0, 3, 6.18879302],
        [0, 4, 4.81736381],
        [1, 0, 14.085559],
        [1, 1, 11.8850568],
        [1, 2, 9.68455459],
        [1, 3, 7.48405236],
        [1, 4, 5.28355014],
        [2, 0, 17.8680374],
        [2, 1, 14.8384622],
        [2, 2, 11.8088869],
        [2, 3, 8.7793117],
        [2, 4, 5.74973647],
        [3, 0, 21.6505158],
        [3, 1, 17.7918675],
        [3, 2, 13.9332193],
        [3, 3, 10.074571],
        [3, 4, 6.21592281],
        [4, 0, 25.4329941],
        [4, 1, 20.7452729],
        [4, 2, 16.0575516],
        [4, 3, 11.3698304],
        [4, 4, 6.68210914],
      ];
      const dataMax = 25.4329941;
      const dataMin = 4.81736381;
      */
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        title: {
          text: 'PDE derivation accuracy',
        },
        tooltip: {},
        xAxis: {
          type: 'category',
          name: 'neurons',
          nameLocation: 'center',
          nameTextStyle: {
            verticalAlign: 'top',
            padding: [4, 4, 4, 4],
          },
          axisLabel: {
            interval: 9,
            showMinLabel: false,
          },
          data: this.xData,
        },
        yAxis: {
          type: 'category',
          data: this.yData,
          name: 'layers',
          nameLocation: 'center',
          nameTextStyle: {
            verticalAlign: 'bottom',
            padding: [4, 4, 4, 4],
          },
          axisLabel: {
            interval: 9,
            showMinLabel: false,
          },
        },
        visualMap: {
          min: this.dataMin2,
          max: this.dataMax2,
          calculable: true,
          realtime: false,
          inRange: {
            color: ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026'],
          },
        },
        series: [{
          type: 'heatmap',
          data: this.dataHeat2,
          emphasis: {
            itemStyle: {
              borderColor: '#333',
              borderWidth: 1,
            },
          },
          progressive: 1000,
          animation: false,
        }],
      });
    },
    getHeatmapData() {
      const path = 'http://localhost:5000/heat';
      axios.get(path)
        .then((res) => {
          this.heatmap = res.data.heatmap;
          this.message = res.data.status;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    changeRegressionRidge() {
      this.regressionType = 'Ridge';
      const payload = {
        regression: 'Ridge',
      };
      this.updateRegression(payload);
    },
    changeRegressionLasso() {
      this.regressionType = 'Lasso';
      const payload = {
        regression: 'Lasso',
      };
      this.updateRegression(payload);
    },
    changeRegressionElasticnet() {
      this.regressionType = 'Elasticnet';
      const payload = {
        regression: 'Elasticnet',
      };
      this.updateRegression(payload);
    },
    calculate() {
      const payload = {
        regression: 'Ridge',
      };
      this.updateRegression(payload);
    },
    drawTransferLinear(id) {
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        title: {
          text: 'Linear',
        },
        grid: {
          x: 40,
          y: 35,
          x2: 10,
          y2: 30,
        },
        xAxis: {
          type: 'category',
          data: this.arr1,
          splitArea: {
            show: true,
          },
          axisLabel: {
            rotate: 50,
            interval: 0,
          },
        },
        yAxis: {
          type: 'category',
          data: this.arr2,
          splitArea: {
            show: true,
          },
        },
        visualMap: {
          min: -20,
          max: 20,
          calculable: true,
          show: false,
        },
        series: [
          {
            name: 'Punch Card',
            type: 'heatmap',
            data: this.transfer_linear,
            label: {
              show: false,
            },
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowColor: 'rgba(0, 0, 0, 0.5)',
              },
            },
          },
        ],
      });
    },
    drawTransferNonlinear(id) {
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        title: {
          text: 'Nonlinear',
        },
        grid: {
          x: 40,
          y: 35,
          x2: 10,
          y2: 30,
        },
        xAxis: {
          type: 'category',
          data: this.arr1,
          splitArea: {
            show: true,
          },
          axisLabel: {
            rotate: 50,
            interval: 0,
          },
        },
        yAxis: {
          type: 'category',
          data: this.arr1,
          splitArea: {
            show: true,
          },
        },
        visualMap: {
          min: -20,
          max: 20,
          calculable: true,
          show: false,
        },
        series: [
          {
            name: 'Punch Card',
            type: 'heatmap',
            data: this.transfer_nonlinear,
            label: {
              show: false,
            },
            emphasis: {
              itemStyle: {
                shadowBlur: 10,
                shadowColor: 'rgba(0, 0, 0, 0.5)',
              },
            },
          },
        ],
      });
    },
    drawCausality(id) {
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        title: { text: 'Causality' },
        // tooltip: {
        //   formatter: function func1(x) {
        //     return x.data.des;
        //   },
        // },
        series: [
          {
            type: 'graph',
            layout: 'force',
            roam: true,
            edgeSymbol: ['circle', 'arrow'],
            edgeSymbolSize: [4, 10],
            symbolSize: 30,
            edgeLabel: {
              normal: {
                show: false,
                textStyle: {
                  fontSize: 10,
                },
                // formatter: function func2(x) {
                //   return x.data.name;
                // },
              },
            },
            force: {
              repulsion: 250,
              edgeLength: [50, 150],
            },
            draggable: true,
            itemStyle: {
              normal: {
                color: '#4b565b',
              },
            },
            lineStyle: {
              normal: {
                width: 2,
                color: '#4b565b',
              },
            },
            label: {
              normal: {
                show: true,
                textStyle: {},
              },
            },
            data: [
              {
                name: 'u_t',
                itemStyle: {
                  normal: {
                    color: 'blue',
                  },
                },
              },
              {
                name: 'u',
              },
              {
                name: 'u_x',
              },
              {
                name: 'u_y',
              },
              {
                name: 'u_z',
              },
              {
                name: 'u_xx',
              },
              {
                name: 'u_yy',
              },
              {
                name: 'u_zz',
              },
              {
                name: 'u_xy',
              },
              {
                name: 'u_xz',
              },
              {
                name: 'u_yz',
              },
            ],
            links: [
              {
                source: 'u_x',
                target: 'u_t',
                name: 0.01,
                lineStyle: {
                  normal: {
                    color: '#000',
                  },
                },
              },
              {
                source: 'u_y',
                target: 'u_t',
                name: 0.01,
                lineStyle: {
                  normal: {
                    color: '#000',
                  },
                },
              },
              {
                source: 'u_z',
                target: 'u_t',
                name: 0.01,
                lineStyle: {
                  normal: {
                    color: '#000',
                  },
                },
              },
              {
                source: 'u_xx',
                target: 'u_t',
                name: 0.01,
                lineStyle: {
                  normal: {
                    color: '#000',
                  },
                },
              },
              {
                source: 'u_yy',
                target: 'u_t',
                name: 0.01,
                lineStyle: {
                  normal: {
                    color: '#000',
                  },
                },
              },
              {
                source: 'u_zz',
                target: 'u_t',
                name: 0.01,
                lineStyle: {
                  normal: {
                    color: '#000',
                  },
                },
              },
            ],
          },
        ],
      });
    },
    getRegression() {
      const path = 'http://localhost:5000/statistics';
      axios
        .get(path)
        .then((res) => {
          this.stat = res.data.statistics;
          this.message = res.data.status;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    updateRegression(payload) {
      const path = 'http://localhost:5000/statistics';
      axios
        .post(path, payload)
        .then(() => {
          this.getRegression();
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getRegression();
        });
    },
  },
  mounted() {
    // this.drawTransferLinear('transfer_linear');
    // this.drawTransferNonlinear('transfer_nonlinear');
    // this.drawCausality('causality');
    this.drawHeatmap('heatmap');
    this.drawHeatmap2('heatmap2');
  },
  watch: {
    stat: {
      handler(newValue, oldValue) {
        if (newValue !== oldValue) {
          this.arr1 = newValue.candidates;
          this.transfer_linear = newValue.transfer_linear;
          this.transfer_nonlinear = newValue.transfer_nonlinear;
          this.arr2 = newValue.arr2;
          this.candidates_left = newValue.candidates_left;
          this.coef = newValue.coef;
          // this.drawTransferLinear('transfer_linear');
          // this.drawTransferNonlinear('transfer_nonlinear');
        }
      },
      deep: true,
    },
    heatmap: {
      handler(newValue, oldValue) {
        if (newValue !== oldValue) {
          this.dataMin = newValue.dataMin;
          this.dataMax = newValue.dataMax;
          this.dataMin2 = newValue.dataMin2;
          this.dataMax2 = newValue.dataMax2;
          this.xData = newValue.xData;
          this.yData = newValue.yData;
          this.dataHeat = newValue.dataHeat;
          this.dataHeat2 = newValue.dataHeat2;
          this.drawHeatmap('heatmap');
          this.drawHeatmap2('heatmap2');
        }
      },
      deep: true,
    },
  },
};
</script>
