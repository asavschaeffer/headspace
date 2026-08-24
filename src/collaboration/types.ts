import type { ReducedContext } from '../kernel/select';
import type { ActorId } from '../kernel/types';

export type CollaboratorFailureCode =
  | 'invalid-completion-request'
  | 'completion-context-too-large'
  | 'unknown-collaborator'
  | 'collaborator-unavailable'
  | 'collaborator-timeout'
  | 'provider-rejected'
  | 'provider-invalid-response';

export interface CollaboratorDiagnostic {
  code: CollaboratorFailureCode;
  phase: 'configure' | 'request' | 'provider' | 'response';
  message: string;
  retryable: boolean;
}

export interface CollaboratorCapability {
  id: string;
  version: string;
  label: string;
  actorId: ActorId;
  execution: 'local' | 'remote';
  proposalOnly: true;
  model?: string;
  availability:
    | { status: 'ready' }
    | { status: 'unavailable'; diagnostic: CollaboratorDiagnostic };
}

export interface CompletionRequest {
  collaboratorId: string;
  instruction: string;
  context: ReducedContext;
}

export interface CompletionResult {
  text: string;
  collaboratorId: string;
  collaboratorVersion: string;
  actorId: ActorId;
  model?: string;
  providerResponseId?: string;
}

