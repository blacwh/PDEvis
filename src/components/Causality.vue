<template>
  <div class='pde-view border border-dark' style='height: 340px'>
    <h5 class='title bg-primary text-white m-0'>Causality View</h5>
    <div class='d-flex flex-inline'>
      <div class='border border-dark' style='width: 369px'>
        <input type="radio" name='causality' @change="changeCCM()">
        <span>Convergent Cross Mapping</span>
      </div>
      <div class='border border-dark' style='width: 368px'>
        <input type="radio" name='causality' @change="changeGC()">
        <span>Granger Causality</span>
      </div>
      <div class='border border-dark' style='width: 369px'>
        <input type="radio" name='causality' @change="changeTE()">
        <span>Transfer Entropy</span>
      </div>
    </div>
    <div class='d-flex justify-content-around'>
      <p class='mb-0 mt-2'>Scatter:</p>
      <div class='d-inline-flex'>
        <small>x:</small>
        <input type='range' id='rangeX' min='0.0' max='1.0' value='0.5' step='0.1'
          @input="changeX()">
        <span id="valueX">1.0</span>
      </div>
      <div class='d-inline-flex'>
        <small>y:</small>
        <input type='range' id='rangeY' min='0.0' max='1.0' value='0.5' step='0.1'
          @input="changeY()">
        <span id="valueY">1.0</span>
      </div>
      <div class='d-inline-flex'>
        <small>z:</small>
        <input type='range' id='rangeZ' min='0.0' max='1.0' value='0.5' step='0.1'
          @input="changeZ()">
        <span id="valueZ">1.0</span>
      </div>
    </div>
    <div class='d-flex flex-row'>
      <div class='d-flex flex-column'>
        <button class='btn btn-light button' @click='refresh()'>Refresh</button>
        <div id='sampling' style='width: 300px; height: 238px'></div>
      </div>
      <div class='d-flex flex-column'>
        <div class='d-flex justify-content-around'>
          <div class='select'>
            <select class="form-control" id="option">
              <!-- <option value='u_x'>dx</option>
              <option value='u_y'>dy</option>
              <option value='u_z'>dz</option>
              <option value='u_xx'>dxx</option>
              <option value='u_xy'>dxy</option>
              <option value='u_xz'>dxz</option>
              <option value='u_yy'>dyy</option>
              <option value='u_yz'>dyz</option>
              <option value='u_zz'>dzz</option>-->
              <option value='u_x'>dx</option>
              <option value='u_y'>dy</option>
              <option value='u_xx'>dxx</option>
              <option value='u_xy'>dxy</option>
              <option value='u_yy'>dyy</option>
              <option value='u_xxx'>dxxx</option>
              <option value='u_xxy'>dxxy</option>
              <option value='u_xyy'>dxyy</option>
              <option value='u_yyy'>dyyy</option>
            </select>
          </div>
          <icon name="arrow-right"></icon>
          <div class='select'>
            <select class="form-control" id="dt">
              <option value='u_t'>dt</option>
            </select>
          </div>
          <button class='btn btn-light button' @click='detect()'>Detect</button>
        </div>
        <div id="smap" style="width: 290px; height: 240px;"></div>
      </div>
      <div class='d-flex flex-column'>
        <div class='d-flex justify-content-around'>
          <button class='btn btn-light button' @click='addTerm()'>Add This Term</button>
        </div>
        <div id="acf" style="width: 260px; height: 240px;"></div>
      </div>
      <div class='d-flex flex-column'>
        <div class='d-flex justify-content-around'>
          <button class='btn btn-light button' @click='deleteTerm()'>Delete This Term</button>
        </div>
        <div id="ccm" style="width: 260px; height: 240px;"></div>
      </div>
    </div>
  </div>
</template>

<script>
import echarts from 'echarts';
import 'echarts-gl';
import axios from 'axios';

export default {
  name: 'PdeView',
  data() {
    return {
      charts: '',
      triData: [],
      predMin: [],
      predMax: [],
      message: '',
      valueX: '',
      valueY: '',
      valueZ: '',
      dataTheta: [],
      dataSmap: [],
      minmae: '',
      maxmae: '',
      dataLibrary: [],
      dataCcm: [],
      type: '',
      causality_info: '',
      objective: '',
      candidates: [],
      dataAcf: [],
    };
  },
  methods: {
    changeCCM() {
      this.type = 'ccm';
    },
    changeGC() {
      this.type = 'gc';
    },
    changeTE() {
      this.type = 'te';
    },
    changeX() {
      this.valueX = document.getElementById('rangeX').value;
      document.getElementById('valueX').innerHTML = this.valueX;
    },
    changeY() {
      this.valueY = document.getElementById('rangeY').value;
      document.getElementById('valueY').innerHTML = this.valueY;
    },
    changeZ() {
      this.valueZ = document.getElementById('rangeZ').value;
      document.getElementById('valueZ').innerHTML = this.valueZ;
    },
    detect() {
      this.objective = document.getElementById('option').value;
      const payload = {
        x: this.valueX,
        y: this.valueY,
        z: this.valueZ,
        type: this.type,
        objective: this.objective,
      };
      this.updateCausality(payload);
    },
    refresh() {
      const payload = {
        type: this.type,
        objective: this.objective,
      };
      this.updateCausalityVolume(payload);
    },
    addTerm() {
      this.objective = document.getElementById('option').value;
      const arr = this.candidates;
      arr.push(this.objective);
      const newArr = [];
      for (let i = 0; i < arr.length; i += 1) {
        if (newArr.indexOf(arr[i]) < 0) {
          newArr.push(arr[i]);
        }
      }
      this.candidates = newArr;
      const payload = {
        candidates: this.candidates,
      };
      this.updateCandidate(payload);
    },
    deleteTerm() {
      this.objective = document.getElementById('option').value;
      const arr = this.candidates;
      const newArr = [];
      for (let i = 0; i < arr.length; i += 1) {
        if (arr[i] !== this.objective) {
          newArr.push(arr[i]);
        }
      }
      this.candidates = newArr;
      const payload = {
        candidates: this.candidates,
      };
      this.updateCandidate(payload);
    },
    draw3DScatter(id) {
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        visualMap: {
          show: false,
          // min: this.predMin,
          // max: this.predMax,
          min: 0,
          max: 1,
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
          left: '5%',
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
    drawSmap(id) {
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        // title: {
        //   text: '',
        // },
        // tooltip: {
        //   trigger: 'axis',
        // },
        // legend: {
        //   top: '10%',
        //   data: ['dx', 'dy', 'dz', 'dxx', 'dyy', 'dzz', 'parameter'],
        // },
        grid: {
          left: '22%',
          right: '0%',
          bottom: '12%',
          top: '15%',
          containLabel: true,
        },
        // toolbox: {
        //   feature: {
        //     saveAsImage: {},
        //   },
        // },
        xAxis: {
          name: 'S-map Localisation',
          nameLocation: 'center',
          type: 'category',
          nameTextStyle: {
            verticalAlign: 'top',
            padding: [4, 4, 4, 4],
          },
          // boundaryGap: false,
          // data: this.dataTheta,
          data: [0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
            1.3,
            1.4,
            1.5,
            1.6,
            1.7,
            1.8,
            1.9,
            2.0,
            2.1,
            2.2,
            2.3,
            2.4,
            2.5,
            2.6,
            2.7,
            2.8,
            2.9,
            3.0],
        },
        yAxis: {
          name: 'Prediction Skill (MAE)',
          nameLocation: 'end',
          type: 'value',
        },
        series: [
          {
            type: 'line',
            data: this.dataSmap,
            // data: [2.9688599900222198e-06,
            //   2.930310389983237e-06,
            //   2.894049812496735e-06,
            //   2.8615334617030367e-06,
            //   2.828626444361266e-06,
            //   2.796992681760178e-06,
            //   2.7699580779522882e-06,
            //   2.741822505907459e-06,
            //   2.717670390109541e-06,
            //   2.699787546788097e-06,
            //   2.684224844471941e-06,
            //   2.674165283429469e-06,
            //   2.6716744181893567e-06,
            //   2.6729711335450524e-06,
            //   2.6749660033762726e-06,
            //   2.680354185656871e-06,
            //   2.689190770754575e-06,
            //   2.6977641596173476e-06,
            //   2.706084072397808e-06,
            //   2.7141578748384166e-06,
            //   2.7233541967710112e-06,
            //   2.7368066718225456e-06,
            //   2.7504044369047235e-06,
            //   2.7638674488745628e-06,
            //   2.7770712687466354e-06,
            //   2.790057272292443e-06,
            //   2.8028632138599262e-06,
            //   2.8155211256109846e-06,
            //   2.8280555491271787e-06,
            //   2.8407596478886843e-06],
            smooth: true,
          },
        ],
      });
    },
    drawAcf(id) {
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        grid: {
          left: '20%',
          right: '4%',
          bottom: '15%',
          top: '15%',
          containLabel: true,
        },
        // toolbox: {
        //   feature: {
        //     saveAsImage: {},
        //   },
        // },
        xAxis: {
          name: 'Lag',
          nameLocation: 'center',
          type: 'category',
          nameTextStyle: {
            verticalAlign: 'top',
            padding: [4, 4, 4, 4],
          },
          // boundaryGap: false,
          // data: this.dataLibrary,
          data: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
        },
        yAxis: {
          name: 'ACF',
          nameLocation: 'end',
          type: 'value',
        },
        series: [
          {
            type: 'scatter',
            symbolSize: 7,
            data: this.dataAcf,
            // data: [[10.0, 8.04],
            //   [8.0, 6.95],
            //   [13.0, 7.58],
            //   [9.0, 8.81],
            //   [11.0, 8.33],
            //   [14.0, 9.96],
            //   [6.0, 7.24],
            //   [4.0, 4.26],
            //   [12.0, 10.84],
            //   [7.0, 4.82],
            //   [5.0, 5.68],
            // ],
            smooth: true,
          },
        ],
      });
    },
    drawCcm(id) {
      this.charts = echarts.init(document.getElementById(id));
      this.charts.setOption({
        grid: {
          left: '20%',
          right: '4%',
          bottom: '15%',
          top: '15%',
          containLabel: true,
        },
        // toolbox: {
        //   feature: {
        //     saveAsImage: {},
        //   },
        // },
        xAxis: {
          name: 'Library Size',
          nameLocation: 'center',
          type: 'category',
          nameTextStyle: {
            verticalAlign: 'top',
            padding: [4, 4, 4, 4],
          },
          // boundaryGap: false,
          data: this.dataLibrary,
          // data: [7.0, 17.0, 27.0, 37.0, 47.0, 57.0, 67.0, 77.0, 87.0, 97.0],
        },
        yAxis: {
          name: 'Correlation',
          nameLocation: 'end',
          type: 'value',
        },
        series: [
          {
            name: 'dx',
            type: 'line',
            data: this.dataCcm,
            // data: [0.0,
            //   0.9292204673391946,
            //   0.9536797289912297,
            //   0.9716308993113876,
            //   0.978750398722067,
            //   0.9857313508058114,
            //   0.9906450458386525,
            //   0.9939890377751166,
            //   0.9959034317623133,
            //   0.9979993399141782],
            smooth: true,
          },
        ],
      });
    },
    getCausality() {
      const path = 'http://localhost:5000/causality';
      axios.get(path)
        .then((res) => {
          this.causality_info = res.data.causality_info;
          this.message = res.data.status;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    updateCausality(payload) {
      const path = 'http://localhost:5000/causality';
      axios.post(path, payload)
        .then(() => {
          this.getCausality();
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getCausality();
        });
    },
    getCausalityVolume() {
      const path = 'http://localhost:5000/causality_volume';
      axios.get(path)
        .then((res) => {
          this.triData = res.data.causality_info.triData;
          this.predMin = res.data.causality_info.triMin;
          this.predMax = res.data.causality_info.triMax;
          this.message = res.data.status;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    updateCausalityVolume(payload) {
      const path = 'http://localhost:5000/causality_volume';
      axios.post(path, payload)
        .then(() => {
          this.getCausalityVolume();
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getCausalityVolume();
        });
    },
    getCandidate() {
      const path = 'http://localhost:5000/causality_candidate';
      axios.get(path)
        .then((res) => {
          this.candidates = res.data.causality_candidates.candidates;
          this.message = res.data.status;
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.error(error);
        });
    },
    updateCandidate(payload) {
      const path = 'http://localhost:5000/causality_candidate';
      axios.post(path, payload)
        .then(() => {
          this.getCandidate();
        })
        .catch((error) => {
          // eslint-disable-next-line
          console.log(error);
          this.getCandidate();
        });
    },
  },
  mounted() {
    this.draw3DScatter('sampling');
    this.drawSmap('smap');
    this.drawCcm('ccm');
    this.drawAcf('acf');
  },
  watch: {
    causality_info: {
      handler(newValue, oldValue) {
        if (newValue !== oldValue) {
          this.dataTheta = this.causality_info.local;
          this.dataSmap = this.causality_info.mae;
          this.minmae = this.causality_info.minmae;
          this.maxmae = this.causality_info.maxmae;
          this.dataLibrary = this.causality_info.libsize;
          this.dataCcm = this.causality_info.cor;
          this.dataAcf = this.causality_info.acf;
          this.drawSmap('smap');
          this.drawCcm('ccm');
          this.drawAcf('acf');
        }
      },
      deep: true,
    },
    triData: {
      handler(newValue, oldValue) {
        if (newValue !== oldValue) {
          this.triData = newValue;
          this.draw3DScatter('sampling');
        }
      },
      deep: true,
    },
  },
};
</script>

<style>
#option, #dt {
  height: 17px;
  width: 60px;
  line-height: 17px;
  padding: 0px;
  text-align: center;
  font-size: 12px;
}
</style>
