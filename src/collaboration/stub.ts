// The deterministic reference adapter. It lives outside the kernel so replacing
// it never changes selection, proposal, or acceptance semantics.

import type { CompletionOutput, ReducedContext } from '../kernel/select';
import type { CollaboratorCapability, CompletionRequest, CompletionResult } from './types';

export const OFFLINE_COLLABORATOR: CollaboratorCapability = {
  id: 'headspace.offline-deterministic',
  version: '1',
  label: 'offline deterministic collaborator',
  actorId: 'agent:stub',
  execution: 'local',
  proposalOnly: true,
  availability: { status: 'ready' },
};

export const stubCompleter = async (context: ReducedContext, instruction: string): Promise<CompletionOutput> => {
  const focus = context.items.find((item) => item.role === 'focus');
  return {
    text:
      `(stub) In response to "${instruction}" — a real model adapter plugs in behind the Completer seam. ` +
      `It saw ${context.items.length} context items (${context.chars} chars), focused on ${focus?.chunkId ?? 'nothing'}.`,
    producer: { id: OFFLINE_COLLABORATOR.id, version: OFFLINE_COLLABORATOR.version },
  };
};

export async function completeWithStub(
  context: ReducedContext,
  instruction: string,
): Promise<CompletionResult> {
  return {
    text: (await stubCompleter(context, instruction)).text,
    collaboratorId: OFFLINE_COLLABORATOR.id,
    collaboratorVersion: OFFLINE_COLLABORATOR.version,
    actorId: OFFLINE_COLLABORATOR.actorId,
  };
}

export async function dispatchToLocalCollaborator(request: CompletionRequest): Promise<CompletionResult> {
  if (request.collaboratorId !== OFFLINE_COLLABORATOR.id) {
    throw new Error(`No local collaborator implementation is registered for ${request.collaboratorId}.`);
  }
  return completeWithStub(request.context, request.instruction);
}
