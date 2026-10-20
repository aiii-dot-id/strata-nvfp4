import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './styles/kit/tokens.css';
import './styles/kit/components.css';
import './styles/app.css';
import { App } from './app/App';
import { installDropGuard } from './components/DropZone';
import { installFork } from './lib/fork';   // strata-nvfp4
import './styles/fork.css';   // strata-nvfp4

const el = document.getElementById('root');
if (!el) throw new Error('#root missing');

installDropGuard();
installFork();   // strata-nvfp4: hides what an attached interface cannot do

createRoot(el).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

import './styles/annotation-fixes.css';
import './styles/monitor.css';
import './styles/time-skip.css';

import './styles/model-run.css';

import './styles/onboarding.css';
