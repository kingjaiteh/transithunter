declare module 'plotly.js-basic-dist-min' {
  import Plotly from 'plotly.js'
  export default Plotly
}

declare module 'react-plotly.js/factory' {
  import type { PlotParams } from 'react-plotly.js'
  import type * as React from 'react'
  export default function createPlotlyComponent(plotly: unknown): React.ComponentType<PlotParams>
}
