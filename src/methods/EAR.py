"""EAR: Equilibrium Aggregation Reasoning.

Generates k candidate reasoning paths, then iteratively critiques them,
judges each critique with a separate model call, and (optionally) lets
successfully-critiqued candidates revise their answers.

The final answer is selected by an equilibrium selector (Elo, Nash, Survival)
that scores candidates based on how well they withstand critique.

All intermediate traces (critiques, judge verdicts, revisions, round scores)
are persisted in the returned dictionary for downstream analysis.
"""

from typing import Any, Dict, List, Optional

from src.methods.Base import BaseReasoningMethod
from src.methods.ear.judge import Judge
from src.methods.ear.types import CritiqueResult, RevisionResult
from src.registry import METHODS


@METHODS.register("ear")
class EAR(BaseReasoningMethod):
    def __init__(
        self,
        model: Any,
        prompt_template: Any,
        parser: Any,
        critique_strategy: Any,
        equilibrium_selector: Any,
        judge: Judge,
        revision_prompt: Any,
        k: int = 5,
        rounds: int = 3,
        allow_revision: bool = True,
    ):
        super().__init__(model, prompt_template, parser)
        self.critique_strategy = critique_strategy
        self.equilibrium_selector = equilibrium_selector
        self.judge = judge
        self.revision_prompt = revision_prompt
        self.k = k
        self.rounds = rounds
        self.allow_revision = allow_revision

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_candidate(self, question: str, candidate_id: int, choices=None, **kwargs: Any) -> Dict:
        # prompt = self.prompt_template.format(question=question)
        if choices is not None:
            prompt = self.prompt_template.format(question=question, choices=choices)
        else:
            prompt = self.prompt_template.format(question=question)
        generation = self.model.generate(prompt, **kwargs)
        extracted_answer, answer_start = self.parser.parse(generation)
        reasoning_text = generation[:answer_start]
        answer_text = generation[answer_start:]
        return {
            "id": candidate_id,
            "prompt": prompt,
            "raw_generation": generation,
            "reasoning_text": reasoning_text,
            "answer_text": answer_text,
            "extracted_answer": extracted_answer,
            "parse_success": extracted_answer is not None,
            "tokens": {
                "total": self.model.count_tokens(generation),
                "reasoning": self.model.count_tokens(reasoning_text),
                "answer": self.model.count_tokens(answer_text),
            },
        }

    def _revise_candidate(
        self,
        question: str,
        candidate: Dict,
        successful_critiques: List[CritiqueResult],
        **kwargs: Any,
    ) -> Optional[RevisionResult]:
        if not successful_critiques:
            return None
        # Use the first successful critique; could be extended to pick strongest
        critique = successful_critiques[0]
        prompt = self.revision_prompt.format(
            question=question,
            reasoning=candidate.get("reasoning_text") or "",
            answer=candidate.get("extracted_answer") or "",
            critique=critique.critique_text,
        )
        revised_raw = self.model.generate(prompt, **kwargs)
        new_answer, answer_start = self.parser.parse(revised_raw)
        reasoning_text = revised_raw[:answer_start]
        answer_text = revised_raw[answer_start:]

        answer_changed = new_answer is not None and new_answer != candidate.get("extracted_answer")

        return RevisionResult(
            candidate_id=candidate["id"],
            old_answer=candidate.get("extracted_answer"),
            new_answer=new_answer,
            revised_raw=revised_raw,
            reasoning_text=reasoning_text,
            answer_text=answer_text,
            tokens=self.model.count_tokens(revised_raw),
            answer_changed=answer_changed,
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, instance: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        question = instance.get("question")
        if not question:
            raise ValueError("Instance must contain a 'question' key.")

        # 1. Generate k initial candidates
        choices = instance.get("choices")
        candidates = [self._generate_candidate(question, i, choices=choices, **kwargs) for i in range(self.k)]

        # Snapshot initial state before any in-place revisions
        initial_snapshot = [
            {
                "id": c["id"],
                "extracted_answer": c["extracted_answer"],
                "parse_success": c["parse_success"],
                "tokens": c["tokens"],
            }
            for c in candidates
        ]

        # 2. Initialize selector
        self.equilibrium_selector.initialize([c["id"] for c in candidates])

        rounds_log = []
        total_critique_tokens = 0
        total_judge_tokens = 0
        total_revision_tokens = 0
        total_model_calls = self.k  # count initial generations
        rounds_until_convergence = self.rounds

        for round_num in range(1, self.rounds + 1):
            # a. Critique
            critiques = self.critique_strategy.run(question, candidates, self.model, **kwargs)
            total_critique_tokens += sum(c.tokens for c in critiques)
            total_model_calls += len(critiques)

            # b. Judge each critique (paired with its critique for easy access in revision)
            judged_pairs = []
            for critique in critiques:
                target = next(c for c in candidates if c["id"] == critique.target_id)
                jr = self.judge.evaluate(question, target, critique, **kwargs)
                judged_pairs.append((critique, jr))
                total_judge_tokens += jr.tokens
                total_model_calls += 1

            judge_results = [jr for _, jr in judged_pairs]

            # c. Update equilibrium scores
            self.equilibrium_selector.update(judge_results)

            # d. Revision (optional)
            revisions: List[RevisionResult] = []
            if self.allow_revision:
                successful_by_target: Dict[int, List[CritiqueResult]] = {}
                for critique, jr in judged_pairs:
                    if jr.is_successful:
                        successful_by_target.setdefault(jr.target_id, []).append(critique)

                for candidate in candidates:
                    successful = successful_by_target.get(candidate["id"], [])
                    if not successful:
                        continue
                    revision = self._revise_candidate(question, candidate, successful, **kwargs)
                    if revision is None:
                        continue
                    revisions.append(revision)
                    total_revision_tokens += revision.tokens
                    total_model_calls += 1
                    # Update candidate in-place with revised state
                    candidate["extracted_answer"] = revision.new_answer
                    candidate["reasoning_text"] = revision.reasoning_text
                    candidate["answer_text"] = revision.answer_text
                    candidate["raw_generation"] = revision.revised_raw
                    candidate["parse_success"] = revision.new_answer is not None

            # Notify selector of revision outcomes (Nash uses this for convergence check)
            self.equilibrium_selector.notify_revisions(revisions)

            rounds_log.append(
                {
                    "round_num": round_num,
                    "critiques": [c.to_dict() for c in critiques],
                    "judge_results": [jr.to_dict() for jr in judge_results],
                    "revisions": [rv.to_dict() for rv in revisions],
                    "scores_after_round": {
                        str(cid): score
                        for cid, score in self.equilibrium_selector.get_scores().items()
                    },
                }
            )

            if self.equilibrium_selector.is_converged():
                rounds_until_convergence = round_num
                break

        # 3. Select winner
        winner_id = self.equilibrium_selector.select_winner_id(candidates)
        winner = next(c for c in candidates if c["id"] == winner_id)

        generation_tokens = sum(c["tokens"]["total"] for c in candidates)
        total_tokens = (
            generation_tokens + total_critique_tokens + total_judge_tokens + total_revision_tokens
        )
        total_successful = sum(
            1 for r in rounds_log for jr in r["judge_results"] if jr["is_successful"]
        )

        return {
            "instance_id": instance.get("id"),
            "dataset": instance.get("dataset"),
            "split": instance.get("split"),
            "question": question,
            "ground_truth": instance.get("answer"),
            "initial_candidates": initial_snapshot,
            "rounds": rounds_log,
            "final_scores": {
                str(cid): score for cid, score in self.equilibrium_selector.get_scores().items()
            },
            "selected_candidate_id": winner_id,
            "extracted_answer": winner.get("extracted_answer"),
            "parse_success": winner.get("parse_success", False),
            "run_config": {
                "k": self.k,
                "rounds": self.rounds,
                "critique_strategy": type(self.critique_strategy).__name__,
                "equilibrium_selector": type(self.equilibrium_selector).__name__,
                "allow_revision": self.allow_revision,
                "max_new_tokens": kwargs.get("max_new_tokens"),
            },
            "metrics": {
                "total_tokens": total_tokens,
                "generation_tokens": generation_tokens,
                "critique_tokens": total_critique_tokens,
                "judge_tokens": total_judge_tokens,
                "revision_tokens": total_revision_tokens,
                "total_model_calls": total_model_calls,
                "rounds_until_convergence": rounds_until_convergence,
                "total_critiques": sum(len(r["critiques"]) for r in rounds_log),
                "successful_critiques": total_successful,
                "total_revisions": sum(len(r["revisions"]) for r in rounds_log),
            },
        }
