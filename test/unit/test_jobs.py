# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for job related runtime functions."""

import random
import time
import warnings
from unittest.mock import patch
import numpy as np
from ddt import ddt, data

from qiskit.providers.exceptions import QiskitBackendNotFoundError
from qiskit.circuit import QuantumCircuit, Parameter

from qiskit_ibm_runtime import RuntimeJobV2
from qiskit_ibm_runtime.constants import API_TO_JOB_ERROR_MESSAGE
from qiskit_ibm_runtime.exceptions import (
    RuntimeJobFailureError,
    RuntimeJobNotFound,
    RuntimeJobMaxTimeoutError,
    RuntimeInvalidStateError,
)
from qiskit_ibm_runtime.quantum_program import QuantumProgram
from qiskit_ibm_runtime.quantum_program.quantum_program import CircuitItem
from qiskit_ibm_runtime.quantum_program.quantum_program_params_converters import (
    QuantumProgramParamsConverter,
    AVAILABLE_CONVERTERS,
)
from qiskit_ibm_runtime.options import ExecutorOptions
from .mock.fake_runtime_client import (
    FailedRuntimeJob,
    FailedRanTooLongRuntimeJob,
    CancelableRuntimeJob,
    BaseFakeRuntimeClient,
)
from ..ibm_test_case import IBMTestCase
from ..decorators import run_cloud_fake
from ..program import run_program
from ..utils import mock_wait_for_final_state


@ddt
class TestRuntimeJob(IBMTestCase):
    """Class for testing runtime jobs."""

    @run_cloud_fake
    def test_run_program(self, service):
        """Test running program."""
        params = {"param1": "foo"}
        job = run_program(service=service, inputs=params)
        self.assertTrue(job.job_id())
        self.assertIsInstance(job, RuntimeJobV2)
        with mock_wait_for_final_state(service, job):
            job.wait_for_final_state()
            self.assertEqual(job.status(), "DONE")
            self.assertTrue(job.result())

    @run_cloud_fake
    def test_run_program_phantom_backend(self, service):
        """Test running on a phantom backend."""
        with self.assertRaises(QiskitBackendNotFoundError):
            _ = run_program(service=service, backend_name="phantom_backend")

    @run_cloud_fake
    def test_run_program_with_custom_runtime_image(self, service):
        """Test running program with a custom image."""
        params = {"param1": "foo"}
        image = "name:tag"
        job = run_program(service=service, inputs=params, image=image)
        self.assertTrue(job.job_id())
        self.assertIsInstance(job, RuntimeJobV2)
        with mock_wait_for_final_state(service, job):
            job.wait_for_final_state()
            self.assertTrue(job.result())
        self.assertEqual(job.status(), "DONE")
        self.assertEqual(job.image, image)

    @run_cloud_fake
    def test_run_program_with_custom_log_level(self, service):
        """Test running program with a custom image."""
        job = run_program(service=service, log_level="DEBUG")
        job_raw = service._get_api_client()._get_job(job.job_id())
        self.assertEqual(job_raw.log_level, "DEBUG")

    @run_cloud_fake
    def test_run_program_failed(self, service):
        """Test a failed program execution."""
        job = run_program(service=service, job_classes=FailedRuntimeJob)
        with mock_wait_for_final_state(service, job):
            job.wait_for_final_state()
            job_result_raw = service._get_api_client().job_results(job.job_id())
            self.assertEqual("ERROR", job.status())
            self.assertEqual(
                API_TO_JOB_ERROR_MESSAGE["FAILED"].format(job.job_id(), job_result_raw),
                job.error_message(),
            )
            with self.assertRaises(RuntimeJobFailureError):
                job.result()

    @run_cloud_fake
    def test_run_program_failed_ran_too_long(self, service):
        """Test a program that failed since it ran longer than maximum execution time."""
        job = run_program(service=service, job_classes=FailedRanTooLongRuntimeJob)
        with mock_wait_for_final_state(service, job):
            job.wait_for_final_state()
            job_result_raw = service._get_api_client().job_results(job.job_id())
            self.assertEqual("ERROR", job.status())
            self.assertEqual(
                API_TO_JOB_ERROR_MESSAGE["CANCELLED - RAN TOO LONG"].format(
                    job.job_id(), job_result_raw
                ),
                job.error_message(),
            )
            with self.assertRaises(RuntimeJobMaxTimeoutError):
                job.result()

    @run_cloud_fake
    def test_cancel_job(self, service):
        """Test canceling a job."""
        job = run_program(service, job_classes=CancelableRuntimeJob)
        time.sleep(1)
        job.cancel()
        self.assertEqual(job.status(), "CANCELLED")
        rjob = service.job(job.job_id())
        self.assertEqual(rjob.status(), "CANCELLED")
        with self.assertRaises(RuntimeInvalidStateError) as exc:
            rjob.result()
        self.assertIn("Job was cancelled", str(exc.exception))

    @run_cloud_fake
    def test_final_result(self, service):
        """Test getting final result."""
        job = run_program(service)
        with mock_wait_for_final_state(service, job):
            result = job.result()
            self.assertTrue(result)

    @run_cloud_fake
    def test_job_status(self, service):
        """Test job status."""
        job = run_program(service)
        time.sleep(random.randint(1, 5))
        self.assertTrue(job.status())

    @run_cloud_fake
    def test_wait_for_final_state(self, service):
        """Test wait for final state."""
        job = run_program(service)
        with mock_wait_for_final_state(service, job):
            job.wait_for_final_state()
        self.assertEqual("DONE", job.status())

    @run_cloud_fake
    def test_delete_job(self, service):
        """Test deleting a job."""
        params = {"param1": "foo"}
        job = run_program(service=service, inputs=params)
        self.assertTrue(job.job_id())
        service.delete_job(job.job_id())
        with self.assertRaises(RuntimeJobNotFound):
            service.job(job.job_id())

    @run_cloud_fake
    def test_instance_limit_warning(self, service):
        """Test emitting a warning if instance usage has been reached."""
        # All relevant fields present, account limit reached.
        instance_usage_msg_1 = {
            "usage_consumed_seconds": 1,
            "usage_limit_seconds": 2,
            "usage_limit_reached": True,
        }
        # All relevant fields present, instance limit reached.
        instance_usage_msg_2 = {
            "usage_consumed_seconds": 3,
            "usage_limit_seconds": 2,
            "usage_limit_reached": True,
        }
        # Missing `usage_limit_seconds`, account limit reached.
        instance_usage_msg_3 = {
            "usage_consumed_seconds": 1,
            "usage_limit_reached": True,
        }

        with patch.object(BaseFakeRuntimeClient, "cloud_usage", return_value=instance_usage_msg_1):
            with self.assertWarnsRegex(UserWarning, r"There is currently no more time available"):
                run_program(service=service)

        with patch.object(BaseFakeRuntimeClient, "cloud_usage", return_value=instance_usage_msg_2):
            with self.assertWarnsRegex(UserWarning, r"This instance has met its usage limit"):
                run_program(service=service)

        with patch.object(BaseFakeRuntimeClient, "cloud_usage", return_value=instance_usage_msg_3):
            with self.assertWarnsRegex(UserWarning, r"There is currently no more time available"):
                run_program(service=service)

    @run_cloud_fake
    @data(*AVAILABLE_CONVERTERS.keys())
    def test_job_inputs_executor_decode(self, schema_version, service):
        """Test job.inputs property decodes executor quantum programs for all schema versions."""
        # Create a parameterized quantum circuit
        theta = Parameter("theta")
        phi = Parameter("phi")
        circuit = QuantumCircuit(2)
        circuit.rx(theta, 0)
        circuit.ry(phi, 1)
        circuit.cx(0, 1)
        circuit.measure_all()

        # Create quantum program and options
        quantum_program = QuantumProgram(
            shots=1024,
            items=[CircuitItem(circuit=circuit, circuit_arguments=np.array([[0.5, 0.3]]))],
        )
        options = ExecutorOptions()
        options.execution.init_qubits = True
        options.execution.rep_delay = 0.001

        # Encode to the specified schema version
        encoded_params = QuantumProgramParamsConverter.encode(
            schema_version, quantum_program, options
        )
        params_dict = encoded_params.model_dump()

        # Create a job with executor program_id
        job = run_program(service=service, program_id="executor")

        # Mock the API response to include our encoded params
        with patch.object(
            service._get_api_client(),
            "job_get",
            return_value={
                "id": job.job_id(),
                "backend": "backend0",
                "state": {"status": "COMPLETED"},
                "program": {"id": "executor"},
                "params": params_dict,
            },
        ):
            inputs = job.inputs

            # Verify the params were decoded
            self.assertIn("quantum_program", inputs)
            self.assertIn("options", inputs)
            self.assertIsInstance(inputs["quantum_program"], QuantumProgram)
            self.assertIsInstance(inputs["options"], ExecutorOptions)
            self.assertEqual(inputs["quantum_program"].shots, 1024)
            self.assertEqual(len(inputs["quantum_program"].items), 1)
            self.assertTrue(inputs["options"].execution.init_qubits)
            self.assertEqual(inputs["options"].execution.rep_delay, 0.001)

    @run_cloud_fake
    def test_job_inputs_executor_decode_failure(self, service):
        """Test job.inputs property handles decode failures gracefully."""
        # Create a job with executor program_id
        job = run_program(service=service, program_id="executor")

        # Mock the API response with invalid params that will fail to decode
        with patch.object(
            service._get_api_client(),
            "job_get",
            return_value={
                "id": job.job_id(),
                "backend": "backend0",
                "state": {"status": "COMPLETED"},
                "program": {"id": "executor"},
                "params": {"invalid": "data", "schema_version": "v0.1"},
            },
        ):
            # Should emit a warning and return the raw params
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                inputs = job.inputs

                # Verify warning was raised
                self.assertEqual(len(w), 1)
                self.assertIn("Unable to convert 'params'", str(w[0].message))

                # Verify raw params are returned
                self.assertIn("invalid", inputs)
                self.assertNotIn("quantum_program", inputs)
                self.assertNotIn("options", inputs)

    @run_cloud_fake
    def test_job_inputs_executor_missing_schema_version(self, service):
        """Test job.inputs property handles missing schema_version gracefully."""
        # Create a job with executor program_id
        job = run_program(service=service, program_id="executor")

        # Mock the API response with params missing schema_version
        with patch.object(
            service._get_api_client(),
            "job_get",
            return_value={
                "id": job.job_id(),
                "backend": "backend0",
                "state": {"status": "COMPLETED"},
                "program": {"id": "executor"},
                "params": {"some_param": "value"},
            },
        ):
            # Should emit a warning and return the raw params
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                inputs = job.inputs

                # Verify warning was raised
                self.assertEqual(len(w), 1)
                self.assertIn("Unable to convert 'params'", str(w[0].message))

                # Verify raw params are returned
                self.assertIn("some_param", inputs)
                self.assertNotIn("quantum_program", inputs)

    @run_cloud_fake
    def test_job_inputs_non_executor(self, service):
        """Test job.inputs property doesn't decode for non-executor programs."""
        # Create a job with a different program_id
        job = run_program(service=service, program_id="sampler")

        # Mock the API response
        with patch.object(
            service._get_api_client(),
            "job_get",
            return_value={
                "id": job.job_id(),
                "backend": "backend0",
                "state": {"status": "COMPLETED"},
                "program": {"id": "sampler"},
                "params": {"pubs": [], "options": {}},
            },
        ):
            inputs = job.inputs

            # Verify params are returned as-is without decoding
            self.assertIn("pubs", inputs)
            self.assertIn("options", inputs)
            self.assertNotIn("quantum_program", inputs)
