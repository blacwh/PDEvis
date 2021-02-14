<template>
  <div class="container p-3">
    <div class='row'>
      <div class='col-12'>
        <InputPanel/>
      </div>
    </div>
    <p></p>
    <div class='row'>
      <div class='col-12'>
        <ModelView/>
      </div>
    </div>
    <p></p>
    <div class='row'>
      <div class='col-12'>
        <CausalityView/>
      </div>
    </div>
    <p></p>
    <div class='row'>
      <div class='col-12'>
        <StatisticsView/>
      </div>
    </div>
    <!-- <div class='row'>
      <div class='col-9'>
         <ModelView/>
         <p></p>
        <CausalityView/>
      </div>
      <div class='col-3'>
        <StatisticsView/>
      </div>
    </div> -->
  </div>
</template>

<style>
</style>

<script>
// @ is an alias to /src
import InputPanel from '@/components/Input.vue';
import ModelView from '@/components/Model.vue';
import CausalityView from '@/components/Causality.vue';
import StatisticsView from '@/components/Statistics.vue';

export default {
  name: 'Home',
  components: {
    InputPanel,
    ModelView,
    CausalityView,
    StatisticsView,
  },
};
</script>
