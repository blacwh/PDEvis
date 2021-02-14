<template>
  <div class='column'>
    <button class='btn btn-light button' @click='refresh()'>Refresh</button>
    <div id="heatmap" style="width: 500px; height: 500px"></div>
  </div>
</template>

<script>
import echarts from 'echarts';
import axios from 'axios';

export default {
  name: 'Heatmap',
  data() {
    return {
      charts: '',
      heatmap: '',
      xData: [],
      yData: [],
      dataMax: '',
      dataMin: '',
      dataHeat: [],
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
        // title: {
        //   text: obj,
        // },
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
  },
  mounted() {
    this.drawHeatmap('heatmap');
  },
  watch: {
    heatmap: {
      handler(newValue, oldValue) {
        if (newValue !== oldValue) {
          this.dataMin = newValue.dataMin;
          this.dataMax = newValue.dataMax;
          this.xData = newValue.xData;
          this.yData = newValue.yData;
          this.dataHeat = newValue.dataHeat;
          this.drawHeatmap('heatmap');
        }
      },
      deep: true,
    },
  },
};
</script>
